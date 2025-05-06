from typing import Optional
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from transformers import AutoTokenizer
from transformers import LlamaTokenizer, LlamaModel, LlamaConfig
from transformers import Qwen2Tokenizer, Qwen2Model, Qwen2Config
from transformers import GPT2Tokenizer, GPT2Model, GPT2Config

from peft import get_peft_model, LoraConfig

from einops import rearrange
from models.embed import TokenEmbedding, PositionalEmbedding, TimeEmbedding

llama_names = ['unsloth/Llama-3.2-1B', 'Qwen/Qwen2.5-1.5B', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B']
gpt_names = ['openai-community/gpt2']

    
# 1. ReprogrammingLayer
# To transform the target embedding into the same dimension as the source embedding
class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_llm, d_keys=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()
        self.n_heads = n_heads
        d_keys = d_keys or (d_model // n_heads)
        
        self.target_norm = nn.LayerNorm(d_model)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)

        self.source_norm = nn.LayerNorm(d_llm)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(self.target_norm(target_embedding)).view(B, L, H, -1)
        source_embedding = self.key_projection(self.source_norm(source_embedding)).view(S, H, -1)
        value_embedding = self.value_projection(self.source_norm(value_embedding)).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)
        out = out.reshape(B, L, -1)
        out = self.out_projection(out)
        return out

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / math.sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding


# 2. Network Based LLM
class LLM2Rec(nn.Module):
    def __init__(self,
                 class_num, 
                 llm_name,
                 d_model,
                 input_dim = 90,
                 token_kernels=[5, 11, 21],
                 n_heads = 8,
                 llm_layers=12,
                 frozen_llm_layer=8,
                 batch_seq_len=2000,
                 dropout = 0.1,
                 add_embed_layer = False, 
                 reprogramming = False,
                ):
        super(LLM2Rec, self).__init__()
        # LLM Configs
        self.llm_name = llm_name
        self.llm_layers = llm_layers
        self.frozen_llm_layer = frozen_llm_layer
        # CSI 
        self.seq_len = batch_seq_len
        self.d_model = d_model
        
        self.n_heads = n_heads

        self.add_embed_layer = add_embed_layer
        self.reprogramming = reprogramming

        # 0. Edit Prompt
        self.be_prompt = (
            f"<|start_prompt|>Task: Determine the specific actions of individuals in a given environment based on WiFi Channel State Information (CSI) signal sequences."      
            f"<|end_prompt|>"
            f"<|start_input|>Doppler spectrogram information of Channel State Information:"
        )

        # 1. Load LLM
        self.load_LLM()
        
        # 2. CSI Process
        if self.add_embed_layer:
            self.time_embed_layer = TimeEmbedding(
                input_dim = input_dim, 
                token_kernels = token_kernels, 
                d_model = d_model, 
                d_llm = self.d_llm, 
                n_heads=self.n_heads,
                llm_name = llm_name,
                dropout = dropout,    
            )
        self.position_embed = PositionalEmbedding(self.d_llm)  # 应该是可训练的

        # 3. Add Extral token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_llm), requires_grad=True)

        # 4. Reprogramming Layer
        if reprogramming:
            self.word_embeddings = self.llm_model.get_input_embeddings().weight # 获得权重
            self.word_embeddings.requires_grad = False
            self.vocab_size = self.word_embeddings.shape[0] # 获得词表大小
            self.num_tokens = 1000 
            self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
            self.reprogramming_layer = ReprogrammingLayer(self.d_llm, self.n_heads, self.d_llm)

        # 5. classification head
        self.feature_head = nn.Sequential(
            nn.Linear(self.d_llm, self.d_llm*4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_llm*4, self.d_llm),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head_layer = nn.Sequential(
            nn.Linear(self.d_llm, self.d_llm*4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_llm*4, self.d_llm),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_llm, class_num),
            nn.Dropout(dropout),
        )
    def load_LLM(self):
        assert self.llm_name in llama_names or self.llm_name in gpt_names, f"LLM model {self.llm_name} is not defined"

        if self.llm_name == 'unsloth/Llama-3.2-1B':
            self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_name)
            self.llm_config = LlamaConfig.from_pretrained(self.llm_name)
            self.llm_config.num_hidden_layers = self.llm_layers
            self.llm_model = LlamaModel.from_pretrained(self.llm_name, config=self.llm_config)
            self.d_llm = self.llm_config.hidden_size
        elif self.llm_name == 'Qwen/Qwen2.5-1.5B':
            self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_name)
            self.llm_config = Qwen2Config.from_pretrained(self.llm_name)
            self.llm_config.num_hidden_layers = self.llm_layers
            self.llm_model = Qwen2Model.from_pretrained(self.llm_name, config=self.llm_config)
            self.d_llm = self.llm_config.hidden_size
        elif self.llm_name == 'openai-community/gpt2':
            self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_name)
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

        for param in self.llm_model.embed_tokens.parameters():
            param.requires_grad = False

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
        
        for param in self.llm_model.wte.parameters():
            param.requires_grad = False

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

    def forward(self, x, return_feature=False):
        """
        parameters:
            x: B, T, C -- It is embedded by TimeEmbedding or origin data
            return_embed: bool
        """
        B, T, C = x.shape

        # 0. Prompt Embeddings
        """
        be_prompt = self.llm_tokenizer(self.be_prompt, return_tensors="pt").input_ids
        be_prompt_embed = self.llm_model.get_input_embeddings()(be_prompt.to(x.device)).expand((B, -1, -1))
        """

        # 1. Time Embedding
        if self.add_embed_layer:
            x = self.time_embed_layer(x)

        # 2. Reprograming
        x_text_embed = None
        if self.reprogramming:
            source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)    
            x_text_embed = self.reprogramming_layer(x, source_embeddings, source_embeddings)               

        x_input = x if x_text_embed is None else x_text_embed
        x_input = torch.cat((x_input , self.cls_token.expand(B, 1, -1)), dim=1)

        # 3. LLM Interaction
        llm_output = self.llm_model(inputs_embeds=x_input, output_hidden_states=True)
        x_output = llm_output.last_hidden_state
        x_features = [self.feature_head(fea[:,:-1]) for fea in llm_output.hidden_states[1:]]
        x_cls_fea = x_output[:,-1]

        # 4. logits
        x_logits = self.head_layer(x_cls_fea)
        return_dict = {'logits': x_logits}
        if return_feature:
            return_dict['features'] = x_features
        return return_dict
    
    def predict(self, x_input):
        out_dict = self.forward(x_input)
        logits = out_dict['logits']
        pred_label = torch.argmax(logits, dim=-1)
        return logits, pred_label
    
def build_time_embed(
        input_dim,
        d_model,
        d_llm,
        llm_name,
        token_kernels=[5, 11, 21],
    ):
    
    time_embed_model = TimeEmbedding(
        input_dim = input_dim,
        token_kernels = token_kernels,
        d_model = d_model,
        d_llm = d_llm,
        llm_name = llm_name,
    )

    return time_embed_model

def build_LLM2Rec(
        class_num,
        llm_name, 
        d_model,
        input_dim = 90,
        token_kernels=[5, 11, 21],
        n_heads=8,
        llm_layers=12,  
        start_layer=0,
        frozen_llm_layer=10,
        batch_seq_len=2000,
        lora=False,
        add_embed_layer=False,
        reprogramming=False, 
    )-> LLM2Rec:

    lm_model = LLM2Rec(
            class_num=class_num,
            llm_name=llm_name,
            d_model=d_model,
            input_dim=input_dim,
            token_kernels=token_kernels,
            n_heads=n_heads,
            llm_layers=llm_layers,
            frozen_llm_layer=frozen_llm_layer,
            batch_seq_len=batch_seq_len,
            add_embed_layer=add_embed_layer,
            reprogramming=reprogramming,
        )
    
    if llm_name in llama_names:
        lm_model.frozen_llama(start_layer)
    else:
        lm_model.frozen_gpt2(start_layer)

    if llm_name in llama_names and lora:
        lm_model.llm_lora()
    return lm_model


    