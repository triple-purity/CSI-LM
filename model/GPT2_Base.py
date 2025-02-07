from typing import Optional
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from transformers import GPT2ForSequenceClassification
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers import BertTokenizer, BertModel
from einops import rearrange
from model.embed import TokenEmbedding, PositionalEmbedding, PatchEmbedding


class GPT2FCLS(nn.Module):
    def __init__(self,
                 num_classes, 
                 token_kernel=3,
                 reduce_ratio = 1,
                 input_dim = 90,
                 d_model = 768,
                 patch_len = 20, 
                 gpt_trans_layer=12,
                 frozen_gpt2_layer=8,
                 batch_seq_len=2000, 
                 dropout = 0.1
                ):
        super(GPT2FCLS, self).__init__()
        self.seq_len = batch_seq_len
        self.gpt_layers = gpt_trans_layer
        self.frozen_gpt2_layer = frozen_gpt2_layer
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.d_model = d_model
        self.patch_len = patch_len
        self.reduce_ratio = reduce_ratio

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
            
        # token_embedding is used for [B,T,C]
        self.token_embedding = TokenEmbedding(
            reduce_dim, 
            d_model, 
            kernel=token_kernel
        )

        # patch_embedding is used for [B,C,T]
        self.patch_embedding = PatchEmbedding(input_dim, d_model, patch_len=patch_len, patch_stride=patch_len)

        self.start_token = nn.Parameter(torch.zeros(1, 1, d_model), requires_grad=True)
        self.stop_token = nn.Parameter(torch.zeros(1, 1, d_model), requires_grad=True)

        self.gpt2 = GPT2Model.from_pretrained('gpt2')
        self.gpt2.h = self.gpt2.h[:self.gpt_layers]

        # classifier head
        self.head_for_class_TS = nn.Sequential(
            nn.Linear(d_model, num_classes),
            nn.Dropout(dropout),
        )
        
        self.head_for_class_ST = nn.Sequential(
            nn.Linear(d_model, num_classes),
            nn.Dropout(dropout),
        )

    def frozen_gpt2(self, start_layer: int = 0):
        end_layer = start_layer + self.frozen_gpt2_layer
        assert end_layer <= self.gpt_layers, "frozen layer should be less than total layer"

        for param in self.gpt2.wpe.parameters():
            param.requires_grad = False

        for i, layer in enumerate(self.gpt2.h):
            if i < end_layer and i >= start_layer:
                for i, (name, param) in enumerate(layer.named_parameters()):
                    if 'ln' in name or 'wpe' in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            else:
                break

    def forward(self, x, mode='TS'):
        assert mode in ['TS', 'ST'], "mode should be TS or ST"
        B, T, C = x.shape

        if mode == 'TS':
            if self.is_reduce_time:
                x = x.permute(0, 2, 1)
                x = self.conv_reduce(x)
                x = x.permute(0, 2, 1)

            x = self.token_embedding(x)
            x = torch.cat((self.start_token.expand(B, 1, -1), x), dim=1)
            x = torch.cat((x, self.stop_token.expand(B, 1, -1)), dim=1)

            outputs = self.gpt2(inputs_embeds=x).last_hidden_state

            outputs = outputs[:,-1]
            outputs = self.head_for_class_TS(outputs)
            return outputs
        else:
            x = x.permute(0, 2, 1)
            x, n_vars = self.patch_embedding(x)
            x = torch.cat((self.start_token.expand(B, 1, -1), x), dim=1)
            x = torch.cat((x, self.stop_token.expand(B, 1, -1)), dim=1)
            outputs = self.gpt2(inputs_embeds=x).last_hidden_state

            outputs = outputs[:, -1]
            outputs = self.head_for_class_ST(outputs)
            return outputs
    
    def predict(self, x, mode='TS'):
        x_logits = self.forward(x, mode=mode)
        pre_labels = torch.argmax(x_logits, dim=1)
        return x_logits, pre_labels


def build_GPT2FCLS(
        num_classes, 
        token_kernel=3,
        reduce_ratio = 1,
        input_dim = 90,
        d_model = 768,
        patch_len = 20,  
        gpt_trans_layer=12,
        start_layer=0,
        frozen_gpt2_layer=8,
        batch_seq_len=2000, 
    ):
    model = GPT2FCLS(
            num_classes=num_classes, 
            token_kernel=token_kernel,
            input_dim=input_dim,
            reduce_ratio = reduce_ratio,
            d_model = d_model,
            patch_len = patch_len,
            gpt_trans_layer=gpt_trans_layer,
            frozen_gpt2_layer=frozen_gpt2_layer,
            batch_seq_len=batch_seq_len,
        )
    model.frozen_gpt2(start_layer)
    return model


    