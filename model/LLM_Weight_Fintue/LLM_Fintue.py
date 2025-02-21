from typing import Optional
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from transformers import BertTokenizer, BertModel
from transformers import LlamaModel, LlamaConfig
from transformers import Qwen2Model, Qwen2Config

from peft import get_peft_model, LoraConfig

from einops import rearrange
from model.embed import TokenEmbedding, PositionalEmbedding, PatchEmbedding

use_llm_names = ['unsloth/llama-3-8B', 'unsloth/Qwen2.5-3B']

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

# 2. ReprogrammingLayer
# To transform the target embedding into the same dimension as the source embedding
class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_llm, d_keys=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
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



# 3. Network Based LLM
class LLM2Rec(nn.Module):
    def __init__(self,
                 num_classes,
                 llm_name,
                 d_model,
                 input_dim = 90,
                 token_kernel=3,
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

        # 1. CSI Process
        self.is_reduce_time = reduce_ratio != 1
        reduce_dim = self.input_dim*(reduce_ratio*2) if self.is_reduce_time else self.input_dim
        if self.is_reduce_time:
            self.conv_reduce = nn.Sequential(
                nn.Conv1d(self.input_dim, reduce_dim, 
                          kernel_size=reduce_ratio, stride=reduce_ratio, 
                          bias=True),
                nn.BatchNorm1d(reduce_dim),
                nn.SiLU(inplace=True),
            )
        # token_embedding is used for [B,T,C]
        self.token_embedding = TokenEmbedding(
            reduce_dim, 
            d_model, 
            kernel=token_kernel
        )
        # patch_embedding is used for [B,C,T]
        self.patch_embedding = PatchEmbedding(d_model, patch_len=patch_len, patch_stride=patch_len)

        # 2. Add Extral token
        self.start_token = nn.Parameter(torch.zeros(1, 1, d_model), requires_grad=True)
        self.stop_token = nn.Parameter(torch.zeros(1, 1, d_model), requires_grad=True)

        # 3. Load LLM
        self.load_LLM()

        # 4. Reprogramming Layer
        self.word_embeddings = self.llm_model.get_input_embeddings().weight # 获得权重
        self.vocab_size = self.word_embeddings.shape[0] # 获得词表大小
        self.num_tokens = 1000 
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
        self.reprogramming_layer = ReprogrammingLayer(self.d_model, self.n_heads, self.d_llm)

        # 5.Classifier Head
        self.head_for_class_TS = nn.Sequential(
            nn.Linear(self.d_llm, num_classes),
            nn.Dropout(dropout),
        )
        
        self.head_for_class_ST = nn.Sequential(
            nn.Linear(self.d_llm, num_classes),
            nn.Dropout(dropout),
        )

    def load_LLM(self):
        if self.llm_name == 'unsloth/Llama-3.2-3B':
            self.llm_config = LlamaConfig.from_pretrained(self.llm_name)
            self.llm_config.num_hidden_layers = self.llm_layers
            self.llm_model = LlamaModel.from_pretrained(self.llm_name, config=self.llm_config)
        elif self.llm_name == 'unsloth/Qwen2.5-3B':
            self.llm_config = Qwen2Config.from_pretrained(self.llm_name)
            self.llm_config.num_hidden_layers = self.llm_layers
            self.llm_model = Qwen2Model.from_pretrained(self.llm_name)
        else:
            raise Exception('LLM model is not defined')
        
        self.d_llm = self.llm_config.hidden_size
    
    def llm_lora(self, lora_r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05):
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_config,
            bias="none",
        )

        self.llm_model = get_peft_model(self.llm_model, lora_config)

    def frozen_llm(self, start_layer:int = 0, frozen_blocks=['self_attn', 'mlp']):
        end_layer = start_layer + self.frozen_llm_layer
        assert end_layer <= self.llm_layers, "frozen layer should be less than total layer"

        for i, layer in enumerate(self.llm_model.layers):
            if i >= start_layer and i < end_layer:
                for name, param in layer.named_parameters():
                    if name.split('.')[0] in frozen_blocks:
                        param.requires_grad = False

    def forward(self, x, mode='TS'):
        assert mode in ['TS', 'ST'], "mode should be TS or ST"
        B, T, C = x.shape

        # 1. Repogramming Layer
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        if mode == 'TS':
            if self.is_reduce_time:
                x = x.permute(0, 2, 1)
                x = self.conv_reduce(x)
                x = x.permute(0, 2, 1)

            x = self.token_embedding(x)
            x = torch.cat((self.start_token.expand(B, 1, -1), x), dim=1)
            x = torch.cat((x, self.stop_token.expand(B, 1, -1)), dim=1)
            x = self.reprogramming_layer(x, source_embeddings, source_embeddings)

            outputs = self.llm_model(inputs_embeds=x).last_hidden_state

            outputs = outputs[:,-1]
            outputs = self.head_for_class_TS(outputs)
            return outputs
        else:
            '''
            CSI Data Shape: [B, C, T]->[B*C, L, Patch_len]->[B*L, C, d_model]
            Calculate CSI in Every Block as:
            1) [B*C, L, d_model]->[B*C, L, d_model] -- LLM Block forward
            2) [B*C, L, d_model]->[B*L, C, d_mdoel] -- Reshape to calculate L-th C channel data
            3) [B*L, C, d_model]->[B*L, C, d_model] -- Transformer Encoder
            4) [B*L, C, d_model]->[B*C, L, d_model] -- Reshape to origin
            '''
            x = x.permute(0, 2, 1)
            x, n_vars = self.patch_embedding(x)
            x = torch.cat((self.start_token.expand(B, 1, -1), x), dim=1)
            x = torch.cat((x, self.stop_token.expand(B, 1, -1)), dim=1)
            
            outputs = self.llm_model(inputs_embeds=x).last_hidden_state

            outputs = outputs[:, -1]
            outputs = self.head_for_class_ST(outputs)
            return outputs
    
    def predict(self, x, mode='TS'):
        x_logits = self.forward(x, mode=mode)
        pre_labels = torch.argmax(x_logits, dim=1)
        return x_logits, pre_labels


def build_LLM2Rec(
        num_classes,
        llm_name, 
        d_model,
        input_dim = 90,
        token_kernel=3,
        reduce_ratio = 1,
        patch_len = 20,
        n_heads=8,
        llm_layers=12,  
        start_layer=0,
        frozen_llm_layer=8,
        batch_seq_len=2000,
        lora=True, 
    ):
    model = LLM2Rec(
            num_classes=num_classes,
            llm_name=llm_name,
            d_model=d_model,
            input_dim=input_dim,
            token_kernel=token_kernel,
            reduce_ratio = reduce_ratio,
            patch_len = patch_len,
            n_heads=n_heads,
            llm_layers=llm_layers,
            frozen_llm_layer=frozen_llm_layer,
            batch_seq_len=batch_seq_len,
        )
    model.frozen_llm(start_layer)
    if lora:
        model.llm_lora()
    return model


    