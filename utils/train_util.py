# 该文件用来存放模型训练相关函数
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score

# 损失函数
def DomainDeception(domain_logits: torch.tensor, domain_labels: torch.tensor):
    domain_num = domain_logits.shape[1]
    soft_domain_logits = F.log_softmax(domain_logits, dim=1)
    domain_labels = F.one_hot(domain_labels, domain_num)
    domain_loss = F.kl_div(soft_domain_logits, domain_labels, reduction='batchmean')
    return domain_loss