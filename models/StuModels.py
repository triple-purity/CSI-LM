import torch
import torch.nn as nn
import torch.nn.functional as F

from models.embed import PositionalEmbedding
from models.LM_Base import TimeEmbedding

# 1. Create TimeModule base Transformer
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
    
class EncoderLayer(nn.Module):
    def __init__(self, embed_size, heads, head_dim=None, dropout=0.):
        super(EncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(embed_size)
        self.attention = MultiHeadAttention(embed_size, heads, head_dim, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.GELU(),
            nn.Linear(embed_size * 4, embed_size),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_size)
    
    def forward(self, x, mask=None):
        x = x + self.attention(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x

class DownLayer(nn.Module):
    def __init__(self, embed_size):
        super(DownLayer, self).__init__()
        self.downlayer = nn.Conv1d(embed_size, embed_size, kernel_size=3, padding=1, stride=2, bias=False)

    def forward(self, x):
        x_permuted = x.permute(0, 2, 1)
        x_downsampled = self.downlayer(x_permuted)
        x_out = x_downsampled.permute(0, 2, 1)
        return x_out

class TimeModule(nn.Module):
    def __init__(self,
                 input_dim,
                 token_kernels,
                 llm_name, 
                 embed_size, 
                 n_heads, 
                 head_dim=None, 
                 num_encoder=4, 
                 dropout=0.1
                ):
        super(TimeModule, self).__init__()

        self.time_embed = TimeEmbedding(
            input_dim = input_dim,
            token_kernels = token_kernels,
            d_model = embed_size,
            d_llm = embed_size,
            n_heads=n_heads,
            llm_name = llm_name,   
        )

        self.position_embed = PositionalEmbedding(embed_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_size), requires_grad=True)

        self.layers = nn.ModuleList()
        for i in range(num_encoder):
            if i < num_encoder-1:
                self.layers.append(EncoderLayer(embed_size, n_heads, head_dim, dropout))
                self.layers.append(DownLayer(embed_size))
            else:
                self.layers.append(EncoderLayer(embed_size, n_heads, head_dim, dropout))

        self.head_layer = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.Dropout(dropout),
        )
    def forward(self, 
                x, 
                decoder_mask=False, 
                mask=None, 
                return_embed = True, 
                return_feature=True
            ):
        if decoder_mask and mask is not None:
            ValueError("mask should be provided when decoder_mask is False")

        x_embed = self.time_embed(x)
        x_input = torch.cat((self.cls_token.expand(x_embed.shape[0], 1, -1), x_embed ), dim=1)
        x_input = self.position_embed(x_input)
        for layer in self.layers:
            if isinstance(layer, EncoderLayer):
                if decoder_mask and mask is None:
                    mask = torch.tril(torch.ones(x_input.shape[1], x_input.shape[1]))
                x_input = layer(x_input, mask)
            else:
                x_input = layer(x_input)
        x_input = x_input[:, 0, :]
        x_logits = self.head_layer(x_input) 

        return_dict = {'logits': x_logits} 
        if return_embed:
            return_dict['embed'] = x_embed
        if return_feature:
            return_dict['feature'] = x_input
        return return_dict
        

# 2. Create CSINet base Conv and Attention 
# It's not used in the experiment.
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=bias),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        return x
    
class AttentionBlock(nn.Module):
    def __init__(self, in_channels, bias=False):
        super(AttentionBlock, self).__init__()
        
        # Channel Attention
        cur_channels = in_channels*2
        self.channel_mlp = nn.Sequential(
            nn.Linear(cur_channels, cur_channels//2, bias=bias),
            nn.LeakyReLU(inplace=True),
            nn.Linear(cur_channels//2, cur_channels, bias=bias),
        )

        # Spatial Attention
        self.spatial_conv = nn.Conv1d(2, 1, kernel_size=5, stride=1, padding=2, bias=bias)
    
    
    def forward(self, x):
        # channel attention
        channel_max = torch.max(x, dim=1,keepdim=True).values
        channel_avg = torch.mean(x, dim=1,keepdim=True)
        channel_att = torch.cat([channel_max, channel_avg], dim=-1)
        channel_max, channel_avg = self.channel_mlp(channel_att).chunk(2, dim=-1)
        x = x * (F.sigmoid(channel_max + channel_avg))

        # spatial attention
        spatial_max = torch.max(x, dim=2,keepdim=True).values
        spatial_avg = torch.mean(x, dim=2,keepdim=True)
        spatial_att = torch.cat([spatial_max, spatial_avg], dim=-1)
        spatial_att = spatial_att.permute(0, 2, 1)
        spatial_att = self.spatial_conv(spatial_att)
        spatial_att = spatial_att.permute(0, 2, 1)
        x = x * (F.sigmoid(spatial_att))
        return x

# **有问题**
class CSINet(nn.Module):
    def __init__(self, num_classes, in_channels, out_channels, unified_len=500, attn_blocks=3, bias=False):
        super(CSINet, self).__init__()
        self.in_channels = in_channels
        self.unified_len = unified_len
        self.out_channels = out_channels
        
        cur_len = unified_len
        self.feature_ext = nn.ModuleList()
        for _ in range(attn_blocks):
            self.feature_ext.append(ConvBlock(in_channels, in_channels, bias=bias))
            self.feature_ext.append(AttentionBlock(in_channels, bias=bias))
            cur_len = (cur_len-1)//2+1
        self.feature_ext.append(ConvBlock(in_channels, in_channels, bias=bias))
        cur_len = (cur_len-1)//2+1

        self.avg_layer = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(cur_len, out_channels)
        )

        self.head = nn.Sequential(
            nn.Linear(out_channels, num_classes),
        )
    
    def forward(self, x):
        for layer in self.feature_ext:
            x = layer(x)
        x = self.avg_layer(x)
        return x

    def logits(self, x):
        feature = self.forward(x)
        return self.head(feature)


if __name__ == '__main__':
    x = torch.randn(2, 500, 90)
    model = CSINet(num_classes=2, in_channels=90, out_channels=512, attn_blocks=3, bias=False)
    y = model(x)
    print(y.shape)