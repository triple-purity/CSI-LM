import torch
import torch.nn as nn
import torch.nn.functional as F

import einops


# 1. RMSNorm Layer for Llama
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
    
    
# 2. Create MultiAttention Layer
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