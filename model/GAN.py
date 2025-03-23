import torch
import torch.nn as nn
from torch.functional import F

from model.LLM_Fintue import build_LLM2Rec

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
        output = self.head(x)
        return output


class CSI_GAN(nn.Module):
    def __init__(self,
                 action_num,
                 domain_num,
                 llm_name, 
                 d_model,
                 input_dim = 90,
                 token_kernels=[5, 11, 21],
                 trans_layer = 4,
                 n_heads=8,
                 llm_layers=12,  
                 start_layer=0,
                 frozen_llm_layer=10,
                 batch_seq_len=2000,
                 lora=False,
                 reprogramming=False,
                ):
        super(CSI_GAN, self).__init__()

        self.action_num = action_num
        self.domain_num = domain_num

        # 1. 特征提取器
        self.feature_extracter = build_LLM2Rec(
                                    llm_name,
                                    d_model,
                                    input_dim = input_dim,
                                    token_kernels=token_kernels,
                                    trans_layer = trans_layer,
                                    n_heads=n_heads,
                                    llm_layers=llm_layers,  
                                    start_layer=start_layer,
                                    frozen_llm_layer=frozen_llm_layer,
                                    batch_seq_len=batch_seq_len,
                                    lora=lora,
                                    reprogramming=reprogramming
                                )
        
        # 2. Action Recognition Net
        self.action_net = RecNet(
            self.action_num, 
            self.feature_extracter.d_llm
        )

        # 3. Domain Recognition Net
        self.domain_net = RecNet(
            self.domain_num, 
            self.feature_extracter.d_llm+self.action_num
        )

    def forward(self, x):
        feature_action = self.feature_extracter(x)
        action_logits = self.action_net(feature_action)
        
        feature_domain = self.feature_extracter(x)
        domain_input = torch.concat([feature_domain, action_logits.detach()], dim=-1)
        domain_logits = self.domain_net(domain_input)

        return action_logits, domain_logits

    def predict(self, x):
        x = self.feature_extracter(x)
        action_logits = self.action_net(x)
        pred_action = torch.argmax(action_logits, dim=-1)
        return action_logits, pred_action