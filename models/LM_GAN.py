import torch
import torch.nn as nn
from torch.functional import F

from models.LM_Base import build_LLM2Rec

class RecNet(nn.Module):
    def __init__(self, num_classes, embed_size, dropout=0.1):
        super(RecNet, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(embed_size, embed_size*4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_size*4, embed_size),
            nn.GELU(),
        )

        self.head = nn.Sequential(
            nn.Linear(embed_size, num_classes),
        )
    def forward(self, x):
        x = self.dense(x)
        x = self.head(x)
        return x


class CSI_GAN(nn.Module):
    def __init__(self,
                 action_num,
                 domain_num,
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
                 add_embed_layer=True,
                 reprogramming=False,
                ):
        super(CSI_GAN, self).__init__()

        self.action_num = action_num
        self.domain_num = domain_num

        # 1. 特征提取器
        self.feature_extracter = build_LLM2Rec(
                                    action_num,
                                    llm_name,
                                    d_model,
                                    input_dim = input_dim,
                                    token_kernels=token_kernels,
                                    n_heads=n_heads,
                                    llm_layers=llm_layers,  
                                    start_layer=start_layer,
                                    frozen_llm_layer=frozen_llm_layer,
                                    batch_seq_len=batch_seq_len,
                                    lora=lora,
                                    add_embed_layer=add_embed_layer,
                                    reprogramming=reprogramming
                                )
        
        # 2. Domain Recognition Net
        self.domain_net = RecNet(
            self.domain_num, 
            self.feature_extracter.d_llm+self.action_num
        )

    def forward(self, x):
        lm_outdict = self.feature_extracter.forward(x, return_feature=True)
        features, action_logits = lm_outdict['features'], lm_outdict['logits']

        domain_input = torch.cat([features, action_logits], dim=-1)
        domain_logits = self.domain_net(domain_input)

        return action_logits, domain_logits

    def predict(self, x):
        lm_outdict = self.feature_extracter(x)
        features, action_logits = lm_outdict['features'], lm_outdict['logits']
        pred_action = torch.argmax(action_logits, dim=-1)
        return action_logits, pred_action