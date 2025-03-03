from typing import Optional
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from transformers import BertTokenizer, BertModel
from transformers import LlamaTokenizer, LlamaModel, LlamaConfig
from transformers import Qwen2Tokenizer, Qwen2Model, Qwen2Config
from transformers import GPT2Tokenizer, GPT2Model, GPT2Config

from peft import get_peft_model, LoraConfig

from einops import rearrange
from model.embed import TokenEmbedding, PositionalEmbedding, PatchEmbedding

llama_names = ['unsloth/Llama-3.2-1B', 'Qwen/Qwen2.5-1.5B', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B']
gpt_names = ['openai-community/gpt2']

# 1. Attention Block as Transformer Encoder
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads, head_dim=None, dropout=0.):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = head_dim or embed_size // heads

        self.values = nn.Linear(self.embed_size, self.head_dim*self.heads, bias=False)
        self.keys = nn.Linear(self.embed_size, self.head_dim*self.heads, bias=False)
        self.queries = nn.Linear(self.embed_size, self.head_dim*self.heads, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, L, _ = x.shape
        
        # Split the embedding into self.heads different pieces
        values = self.values(x).reshape(B, L, self.heads, -1).contiguous()
        keys = self.keys(x).reshape(B, L, self.heads, -1).contiguous()
        queries = self.queries(x).reshape(B, L, self.heads, -1).contiguous()

        # Einsum does matrix multiplication for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just a way to do batch matrix multiplication
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        # Mask padded indices so their weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            B, L, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return self.dropout(out)
    
class Encoder(nn.Module):
    def __init__(self, embed_size, heads, head_dim=None, dropout=0.):
        super(Encoder, self).__init__()
        self.norm1 = nn.LayerNorm(embed_size)
        self.attention = MultiHeadAttention(embed_size, heads, head_dim, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.GELU(),
            nn.Linear(embed_size * 4, embed_size),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_size)
    
    def forward(self, x):
        x += self.attention(self.norm1(x))
        x += self.mlp(self.norm2(x))
        return x


# 2. ReprogrammingLayer
# To transform the target embedding into the same dimension as the source embedding
class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_llm, d_keys=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()
        self.n_heads = n_heads
        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / math.sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


# 3. Network Based LLM
class LLM2Rec(nn.Module):
    def __init__(self,
                 num_classes,
                 llm_name,
                 d_model,
                 input_dim = 90,
                 token_kernels=[3, 11, 21],
                 reduce_ratio = 1,
                 patch_len = 20,
                 n_heads = 8,
                 llm_layers=12,
                 frozen_llm_layer=8,
                 batch_seq_len=2000, 
                 dropout = 0.1
                ):
        super(LLM2Rec, self).__init__()
        # LLM Configs
        self.llm_name = llm_name
        self.llm_layers = llm_layers
        self.frozen_llm_layer = frozen_llm_layer
        # CSI 
        self.num_classes = num_classes
        self.seq_len = batch_seq_len
        self.input_dim = input_dim
        self.d_model = d_model
        
        self.patch_len = patch_len
        self.reduce_ratio = reduce_ratio
        self.n_heads = n_heads

        # 0. Edit Prompt
        be_prompt = (
            f"<|start_prompt|>Role: An expert with profound knowledge and extensive practical experience in the field of communications. "
            f"Task: Determine the specific actions of individuals in a given environment based on WiFi Channel State Information (CSI) signal sequences."       
            # f"You need to comprehensively consider various factors such as the propagation characteristics and attenuation patterns of WiFi signals in different environments and under various action states of individuals."
            f"<|end_prompt|>"
            f"<|start_input|>Information of CSI Signals:"
        )
        af_prompt = (
            f"<|end_output|>"
            f"<|start_output|>The current action of the character is"
        )
        be_prompt = self.llm_tokenizer(be_prompt, return_tensors="pt").input_ids
        af_prompt = self.llm_tokenizer(af_prompt, return_tensors="pt").input_ids

        # 1. Load LLM
        self.load_LLM()
        
        # 2. CSI Process
        self.is_reduce_time = reduce_ratio != 1
        reduce_dim = self.input_dim*(reduce_ratio*2) if self.is_reduce_time else self.input_dim
        if self.is_reduce_time:
            self.conv_reduce = nn.Sequential(
                nn.Conv1d(self.input_dim, reduce_dim, 
                          kernel_size=reduce_ratio, stride=reduce_ratio, 
                          bias=True),
                nn.BatchNorm1d(reduce_dim),
                nn.ReLU(inplace=True),
            )
        # 2.1 token_embedding is used for [B,T,C]
        self.token_embeddings = nn.ModuleList(
            [TokenEmbedding(reduce_dim, d_model, token_kernel) for token_kernel in token_kernels]
        )
        self.token_linear = nn.Sequential(
            nn.Linear(d_model*len(token_kernels), self.d_llm),
            nn.Dropout(dropout),
            RMSNorm(self.d_llm) if self.llm_name == 'llama' else nn.LayerNorm(self.d_llm) 
        )
        # 2.2 patch_embedding is used for [B,C,T]
        self.patch_embedding = PatchEmbedding(d_model, patch_len=patch_len, patch_stride=patch_len//2)
        self.path_encoder = Encoder(self.d_model, self.n_heads)

        # 3. Add Extral token
        self.start_token = nn.Parameter(torch.zeros(1, 1, d_model), requires_grad=True)
        self.stop_token = nn.Parameter(torch.zeros(1, 1, d_model), requires_grad=True)

        # 4. Reprogramming Layer
        self.word_embeddings = self.llm_model.get_input_embeddings().weight # 获得权重
        self.word_embeddings.requires_grad = False
        self.vocab_size = self.word_embeddings.shape[0] # 获得词表大小
        self.num_tokens = 1000 
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
        self.reprogramming_layer = ReprogrammingLayer(self.d_model, self.n_heads, self.d_llm)

        # 5.Classifier Head
        self.head = nn.Sequential(
            nn.Linear(self.d_llm, self.num_classes),
            nn.Dropout(dropout),
        )

    def load_LLM(self):
        assert self.llm_name in llama_names or self.llm_name in gpt_names, f"LLM model {self.llm_name} is not defined"

        if self.llm_name == 'unsloth/Llama-3.2-1B':
            self.llm_tokenizer = LlamaTokenizer.from_pretrained(self.llm_name)
            self.llm_config = LlamaConfig.from_pretrained(self.llm_name)
            self.llm_config.num_hidden_layers = self.llm_layers
            self.llm_model = LlamaModel.from_pretrained(self.llm_name, config=self.llm_config)
            self.d_llm = self.llm_config.hidden_size
        elif self.llm_name == 'Qwen/Qwen2.5-1.5B':
            self.llm_tokenizer = Qwen2Tokenizer.from_pretrained(self.llm_name)
            self.llm_config = Qwen2Config.from_pretrained(self.llm_name)
            self.llm_config.num_hidden_layers = self.llm_layers
            self.llm_model = Qwen2Model.from_pretrained(self.llm_name, config=self.llm_config)
            self.d_llm = self.llm_config.hidden_size
        elif self.llm_name == 'openai-community/gpt2':
            self.llm_tokenizer = GPT2Tokenizer.from_pretrained(self.llm_name)
            self.llm_config = GPT2Config.from_pretrained(self.llm_name)
            self.llm_config.n_layer = self.llm_layers
            self.llm_model = GPT2Model.from_pretrained(self.llm_name, config=self.llm_config)
            self.d_llm = self.llm_config.n_embd
        else:
            raise Exception('LLM model is not defined')
        
    def llm_lora(self, lora_r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05):
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
        )

        self.llm_model = get_peft_model(self.llm_model, lora_config)

    def frozen_llama(self, start_layer:int = 0, frozen_blocks=['self_attn', 'mlp']):
        end_layer = start_layer + self.frozen_llm_layer
        assert end_layer <= self.llm_layers, "frozen layer should be less than total layer"

        for i, layer in enumerate(self.llm_model.layers):
            if i >= start_layer and i < end_layer:
                for name, param in layer.named_parameters():
                    if name.split('.')[0] in frozen_blocks:
                        param.requires_grad = False
            else:
                break
    
    def frozen_gpt2(self, start_layer: int = 0):
        end_layer = start_layer + self.frozen_llm_layer
        assert end_layer <= self.llm_layers, "frozen layer should be less than total layer"
        
        for param in self.llm_model.wpe.parameters():
            param.requires_grad = True

        for i, layer in enumerate(self.llm_model.h):
            if i < end_layer and i >= start_layer:
                for i, (name, param) in enumerate(layer.named_parameters()):
                    if 'ln' in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            else:
                break

    def forward(self, x, reprogramming=False):
        B, T, C = x.shape

        # 1. Reduce Time
        if self.is_reduce_time:
            x = x.permute(0, 2, 1)
            x = self.conv_reduce(x)
            x = x.permute(0, 2, 1)

        # 2. Token Embedding
        conv_x = []
        for layer in self.token_embeddings:
            conv_x.append(layer(x))
        x1 = torch.cat(conv_x, dim=-1)
        x1 = self.token_linear(x1)
        if reprogramming:
            source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)    
            x1 = x1+self.reprogramming_layer(x1, source_embeddings, source_embeddings)               
        
        # x1 = torch.cat((x1, self.stop_token.expand(B, 1, -1)), dim=1)
        
        x1 = self.llm_model(inputs_embeds=x1).last_hidden_state
        x1 = x1[:,-1]

        return self.head(x1)

        # 3. Patch Embedding
        '''
        x2 = x.permute(0, 2, 1)
        x2, nvars = self.patch_embedding(x2)
        x2 = self.llm_model(inputs_embeds=x2).last_hidden_state
        x2 = x2.view(B, nvars, -1)
        x2 = torch.cat((self.start_token.expand(B, 1, -1), x2), dim=1)
        x2 = self.path_encoder(x2)[:, 0]
        '''

        '''
        The Extral Method
        CSI Data Shape: [B, C, T]->[B*C, L, Patch_len]->[B*L, C, d_model]
        Calculate CSI in Every Block as:
        1) [B*C, L, d_model]->[B*C, L, d_model] -- LLM Block forward
        2) [B*C, L, d_model]->[B*L, C, d_mdoel] -- Reshape to calculate L-th C channel data
        3) [B*L, C, d_model]->[B*L, C, d_model] -- Transformer Encoder
        4) [B*L, C, d_model]->[B*C, L, d_model] -- Reshape to origin
        '''
    def predict(self, x, reprogramming=False):
        x_logits = self.forward(x, reprogramming=reprogramming)
        pre_labels = torch.argmax(x_logits, dim=1)
        return x_logits, pre_labels


def build_LLM2Rec(
        num_classes,
        llm_name, 
        d_model,
        input_dim = 90,
        token_kernels=[3, 11, 31],
        reduce_ratio = 1,
        patch_len = 20,
        n_heads=8,
        llm_layers=12,  
        start_layer=0,
        frozen_llm_layer=10,
        batch_seq_len=2000,
        lora=False, 
    ):
    model = LLM2Rec(
            num_classes=num_classes,
            llm_name=llm_name,
            d_model=d_model,
            input_dim=input_dim,
            token_kernels=token_kernels,
            reduce_ratio = reduce_ratio,
            patch_len = patch_len,
            n_heads=n_heads,
            llm_layers=llm_layers,
            frozen_llm_layer=frozen_llm_layer,
            batch_seq_len=batch_seq_len,
        )
    if llm_name in llama_names:
        model.frozen_llama(start_layer)
    else:
        model.frozen_gpt2(start_layer)

    if llm_name in llama_names and lora:
        model.llm_lora()
    return model


    