import torch
import torch.nn as nn
import torch.nn.functional as F

from models.embed import PositionalEmbedding
from models.LM_Base import TimeEmbedding
from models.Layers import MultiHeadAttention
    
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
                 class_num,
                 input_dim,
                 token_kernels,
                 llm_name,
                 d_model, 
                 embed_size, 
                 n_heads, 
                 head_dim=None, 
                 num_encoder=4, 
                 dropout=0.1,
                 pos_learn=False,
                ):
        super(TimeModule, self).__init__()

        self.time_embed = TimeEmbedding(
            input_dim = input_dim,
            token_kernels = token_kernels,
            d_model = d_model,
            d_llm = embed_size,
            n_heads=n_heads,
            llm_name = llm_name,   
        )

        self.position_embed = PositionalEmbedding(embed_size, learnable=pos_learn)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_size), requires_grad=True)

        self.layers = nn.ModuleList()
        for i in range(num_encoder):
            self.layers.append(EncoderLayer(embed_size, n_heads, head_dim, dropout))

        self.norm_layer = nn.LayerNorm(embed_size)

        # head layer
        self.feature_head = nn.Sequential(
            nn.Linear(embed_size, embed_size*4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_size*4, embed_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head_layer = nn.Sequential(
            nn.Linear(embed_size, embed_size*4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_size*4, embed_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_size, class_num),
        )
    def forward(self, 
                x, 
                decoder_mask = False, 
                return_embed = False, 
                return_feature = False
            ):

        x_embed = self.time_embed(x)
        if decoder_mask:
            x_input = torch.cat((x_embed, self.cls_token.expand(x_embed.shape[0], 1, -1)), dim=1)
        else:
            x_input = torch.cat((self.cls_token.expand(x_embed.shape[0], 1, -1), x_embed), dim=1)
        x_input = self.position_embed(x_input)

        hidden_feas = []
        for layer in self.layers:
            if isinstance(layer, EncoderLayer):
                if decoder_mask:
                    mask = torch.tril(torch.ones(x_input.shape[1], x_input.shape[1])).cuda()
                x_input = layer(x_input, mask)
                hidden_feas.append(x_input[:,:-1])
            else:
                x_input = layer(x_input)
                hidden_feas.append(x_input[:,1:])
        x_input = self.norm_layer(x_input)
        
        if decoder_mask:
            x_cls_fea = x_input[:,-1,:]
        else:
            x_cls_fea = x_input[:, 0, :]
        x_logits = self.head_layer(x_cls_fea) 

        return_dict = {'logits': x_logits} 
        if return_embed:
            return_dict['embeds'] = x_embed
        if return_feature:
            return_dict['features'] = hidden_feas
        return return_dict
    
    def predict(self, x, decoder_mask=False):
        return_dict = self.forward(x, decoder_mask=decoder_mask)
        action_logits = return_dict['logits']
        pre_labels = torch.argmax(action_logits, dim=-1)
        return action_logits, pre_labels


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

# 
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