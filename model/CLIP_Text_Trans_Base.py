import sys
from typing import List, Optional, Tuple 
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig

import clip
from clip.model import Transformer

from model.embed import TokenEmbedding, PositionalEmbedding


class CNN_MaxPool(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 kernel_filters: List[int]= [10, 20, 40],   
                 bias: bool = False, 
                 dropout: float = 0.2,
                ):
        super(CNN_MaxPool, self).__init__()
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels, kernel_filters[i], kernel_size=kernel_filters[i], stride=1, bias=bias)
            for i in range(len(kernel_filters))
        ])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = x.transpose(-1, -2)
        x_pool = []
        for layer in self.conv_layers:
            r_x = layer(x)
            r_x = F.relu(r_x, inplace=True)
            cur_time = r_x.shape[-1]
            r_x = F.max_pool1d(r_x, kernel_size=cur_time)
            r_x = r_x.squeeze(-1)
            x_pool.append(r_x)
        x_pool = torch.cat(x_pool, dim=1)
        x_pool = self.dropout(x_pool)
        x_pool = F.relu(x_pool)
        return x_pool


class CLIPFCLS(nn.Module):
    def __init__(self, 
                 num_classes, 
                 input_dim = 90, 
                 token_kernel = 3,
                 reduce_ratio = 1,
                 transformer_width = 512,
                 transformer_layers = 12,
                 transformer_heads = 8,
                 batch_seq_len = 2000,
                 init_model_name = "ViT-B/32", 
                ):
        super(CLIPFCLS, self).__init__()
        self.num_classes = num_classes
        self.time_length = batch_seq_len
        self.input_dim = input_dim
        self.init_model_name = init_model_name

        # Process
        self.is_reduce_time = reduce_ratio != 1
        reduce_dim = self.input_dim*(reduce_ratio*2) if self.is_reduce_time else self.input_dim
        if self.is_reduce_time:
            self.conv_reduce = nn.Sequential(
                nn.Conv1d(self.input_dim, reduce_dim, 
                          kernel_size=reduce_ratio, stride=reduce_ratio, 
                          bias=True),
                nn.BatchNorm1d(reduce_dim),
                nn.ReLU(),
            )

        # CSI Embedding
        self.csi_token_embedding = TokenEmbedding(reduce_dim, transformer_width, kernel=token_kernel)

        # Use Attention to Understand CSI
        self.start_token = nn.Parameter(torch.zeros(1, 1, transformer_width), requires_grad=True)
        self.stop_token = nn.Parameter(torch.zeros(1, 1, transformer_width), requires_grad=True)
        self.positional_embedding = PositionalEmbedding(transformer_width)
        self.positional_dropout = nn.Dropout(0.1)

        self.transformer = Transformer(
            width=transformer_width, 
            layers=transformer_layers, 
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(self.time_length//reduce_ratio+2)
        )
        self.trans_act = nn.GELU()
        self.ln_final = nn.LayerNorm(transformer_width)

        self.head_for_class = nn.Sequential(
            nn.Linear(transformer_width, self.num_classes)
        )
        
        self.initialize_parameters()
    def build_attention_mask(self, time_length):
        mask = torch.empty(time_length, time_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def initialize_parameters(self):
        model, _ = clip.load(self.init_model_name)
        model_transformer_state = model.transformer.state_dict()
        self.transformer.load_state_dict(model_transformer_state)
    
    def freeze_params(self, start_block=0, transformer_frozen_blocks=12, freeze_layer=['attn', 'mlp']):
        end_block = start_block + transformer_frozen_blocks
        assert end_block<=12
        for i, block in enumerate(self.transformer.resblocks):
            if i < end_block and i>=start_block:
                for name, layer in block.named_children():
                    if name in freeze_layer:
                        for param in layer.parameters():
                            param.requires_grad = False
    
    def forward(self, x: torch.Tensor):
        bt, _, _ = x.shape
        # process to reduce timestep
        if self.is_reduce_time:
            x = x.permute(0, 2, 1)
            x = self.conv_reduce(x)
            x = x.permute(0, 2, 1)

        x = self.csi_token_embedding(x)
        
        # concat start and stop token and add positional embedding
        x = torch.cat((self.start_token.expand(bt, 1, -1), x), dim=1)
        x = torch.cat((x, self.stop_token.expand(bt, 1, -1)), dim=1)
        x = x + self.positional_embedding(x)
        x = self.positional_dropout(x)
        
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        
        x = self.trans_act(x)
        x = self.ln_final(x)
        x = x[:,-1]
        x_logits = self.head_for_class(x)
        return x_logits

    def predict(self, x):
        x_logits = self.forward(x)
        pre_labels = torch.argmax(x_logits, dim=1)
        return x_logits, pre_labels
    
def build_clip_csi_model(
        num_classes: int, 
        time_length: int=2000,
        token_kernel: int=3, 
        input_dim = 90,
        reduce_ratio = 1,
        start_block: int=0,
        transformer_frozen_blocks: int=12
    ):
    model = CLIPFCLS(
            num_classes=num_classes, 
            time_length=time_length,
            token_kernel=token_kernel,
            input_dim=input_dim,
            reduce_ratio = reduce_ratio,
        )
    
    model.freeze_params(start_block, transformer_frozen_blocks)
    return model
