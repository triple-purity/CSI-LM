import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math

from models.Layers import MultiHeadAttention, RMSNorm 

llama_names = ['unsloth/Llama-3.2-1B', 'Qwen/Qwen2.5-1.5B', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B']
gpt_names = ['openai-community/gpt2']

class PositionalEmbedding(nn.Module):
    def __init__(self, c_out, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, c_out).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, c_out, 2).float()
                    * -(math.log(10000.0) / c_out)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, c_out, kernel=3, stride=1, padding=1):
        super(TokenEmbedding, self).__init__()

        # Local Time Series Attention
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=c_out, kernel_size=kernel, 
                                   stride=stride, padding=padding, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x
    

class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, patch_stride, dropout=0.):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, patch_stride))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = TokenEmbedding(patch_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        output = self.value_embedding(x)
        return output, n_vars
    
# 4. Time Series Embedding
class TimeEmbedding(nn.Module):
    def __init__(self, input_dim, token_kernels, d_model, d_llm, n_heads, llm_name, dropout=0.1):
        super(TimeEmbedding, self).__init__()

        self.input_dim = input_dim
        self.token_kernels = token_kernels
        self.d_model = d_model
        self.d_llm = d_llm
        self.n_heads = n_heads

        # token_embedding is used for [B,T,C]
        # Multi Scale CNN + TokenEmbedding 
        kernel_dims = [(((input_dim*kernel)//2),kernel) for kernel in token_kernels]
        self.token_embeddings = nn.ModuleList(
            [TokenEmbedding(input_dim, dim, kernel, padding=(kernel-1)//2) for dim, kernel in kernel_dims]
        )
        self.token_linear = nn.Sequential(
            nn.Linear(sum([item for item,_ in kernel_dims]), self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            RMSNorm(self.d_model) if llm_name in llama_names else nn.LayerNorm(self.d_llm) 
        )
        self.token_embed = TokenEmbedding(self.d_model, self.d_llm, kernel=9, stride=4, padding=4)
        self.atten_embed = MultiHeadAttention(self.d_llm, self.n_heads, dropout=dropout)
    
    def forward(self, x):
        # 1. Token Embedding
        conv_x = []
        for layer in self.token_embeddings:
            conv_x.append(layer(x))
        x_cat = torch.cat(conv_x, dim=-1)
        x_cat = self.token_linear(x_cat)
        x_cat = self.token_embed(x_cat)
        x_cat = self.atten_embed(x_cat)
        return x_cat