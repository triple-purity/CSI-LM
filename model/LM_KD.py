import torch
import torch.nn as nn
import torch.nn.functional as F

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
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class DowmLayer(nn.Module):
    def __init__(self, embed_size):
        super(DowmLayer, self).__init__()
        self.downlayer = nn.Conv1d(embed_size, embed_size, kernel_size=3, padding=1, stride=2, bias=False)

    def forward(self, x):
        x_permuted = x.permute(0, 2, 1)
        x_downsampled = self.downlayer(x_permuted)
        x_out = x_downsampled.permute(0, 2, 1)
        return x_out

class TimeEncoder(nn.Module):
    def __init__(self, embed_size, heads, head_dim=None, num_encoder=4, dropout=0.1):
        super(TimeEncoder, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_encoder):
            if i < num_encoder-1:
                self.layers.append(Encoder(embed_size, heads, head_dim, dropout))
                self.layers.append(DowmLayer(embed_size))
            else:
                self.layers.append(Encoder(embed_size, heads, head_dim, dropout))
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# 2. Create Small Module 
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

        self.feature_ext = nn.ModuleList()
        for _ in range(attn_blocks):
            self.feature_ext.append(ConvBlock(in_channels, in_channels, bias=bias))
            self.feature_ext.append(AttentionBlock(in_channels, bias=bias))
        self.feature_ext.append(ConvBlock(in_channels, in_channels, bias=bias))

        self.avg_layer = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(in_channels, out_channels)
        )

        self.head = nn.Sequential(
            nn.Linear(in_channels, num_classes),
        )
    
    def forward(self, x):
        for layer in self.feature_ext:
            x = layer(x)
        x = self.avg_layer(x)
        x = F.linear()
        return x

    def logits(self, x):
        feature = self.forward(x)
        return self.head(feature)


if __name__ == '__main__':
    x = torch.randn(2, 500, 90)
    model = CSINet(num_classes=2, in_channels=90, out_channels=512, attn_blocks=3, bias=False)
    y = model(x)
    print(y.shape)