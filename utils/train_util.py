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
    domain_loss = torch.sum(soft_domain_logits * domain_labels) / domain_labels.shape[0]
    return domain_loss

# 概率分布自信度约束
def confidence_loss(logits: torch.tensor, eps=1e-12):
    soft_logits = F.softmax(logits, dim=-1)
    soft_logits = torch.clamp(soft_logits, min=eps, max=1-eps)
    one_soft_logits = 1-soft_logits
    confidence_loss = -torch.mean(torch.log(soft_logits)+torch.log(one_soft_logits))
    return confidence_loss