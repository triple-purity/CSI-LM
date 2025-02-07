from typing import Optional
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from transformers import Qwen2Model, AutoModelForCausalLM
from einops import rearrange
from model.embed import TokenEmbedding, PositionalEmbedding


class Qwen2FCLS(nn.Module):
    def __init__(self,
                num_classes, 
                token_kernel=3,
                reduce_ratio = 1,
                input_dim = 90,
                d_model = 2048, 
                Qwen_trans_layer=36,
                frozen_gpt2_layer=8,
                batch_seq_len=2000, 
                dropout = 0.1 
            ):
        super(Qwen2FCLS, self).__init__()
        self.seq_len = batch_seq_len
        self.Qwen_trans_layers = Qwen_trans_layer
        self.frozen_gpt2_layer = frozen_gpt2_layer
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.d_model = d_model

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
        self.token_embedding = TokenEmbedding(
            reduce_dim, 
            d_model, 
            kernel=token_kernel
        )

        self.start_token = nn.Parameter(torch.zeros(1, 1, d_model), requires_grad=True)
        self.stop_token = nn.Parameter(torch.zeros(1, 1, d_model), requires_grad=True)

        self.llm_model = Qwen2Model.from_pretrained('unsloth/Qwen2.5-3B')
        self.llm_model.layers = self.llm_model.layers[:self.Qwen_trans_layers]

        self.head_for_class = nn.Sequential(
            nn.Linear(d_model, num_classes)
        )

    def frozen_gpt2(self, start_layer: int = 0):
        end_layer = start_layer + self.frozen_gpt2_layer
        assert end_layer <= self.Qwen_trans_layers, "frozen layer should be less than total layer"

        '''
        需要修改
        '''
        for i, layer in enumerate(self.llm_model.layers):
            if i < end_layer and i >= start_layer:
                for i, (name, param) in enumerate(layer.named_parameters()):
                    if 'ln' in name or 'wpe' in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            else:
                break

    def forward(self, x):
        B, _, _ = x.shape

        if self.is_reduce_time:
            x = x.permute(0, 2, 1)
            x = self.conv_reduce(x)
            x = x.permute(0, 2, 1)

        x = self.token_embedding(x)
        x = torch.cat((self.start_token.expand(B, 1, -1), x), dim=1)
        x = torch.cat((x, self.stop_token.expand(B, 1, -1)), dim=1)

        outputs = self.llm_model(inputs_embeds=x).last_hidden_state

        outputs = outputs[:,-1]
        outputs = self.head_for_class(outputs)
        return outputs
    
    def predict(self, x):
        x_logits = self.forward(x)
        pre_labels = torch.argmax(x_logits, dim=1)
        return x_logits, pre_labels