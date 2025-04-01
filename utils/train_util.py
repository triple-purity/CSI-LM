# 该文件用来存放模型训练相关函数
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score

# 1.概率分布自信度约束
def confidence_loss(logits: torch.tensor, eps=1e-12):
    soft_logits = F.softmax(logits, dim=-1)
    soft_logits = torch.clamp(soft_logits, min=eps, max=1-eps)
    one_soft_logits = 1-soft_logits
    confidence_loss = -torch.mean(torch.log(soft_logits)+torch.log(one_soft_logits))
    return confidence_loss

# 2. Label-InfoCE ---- Contrastive Learning
def InfoCE(features: torch.tensor, labels: torch.tensor, eps=1e-12):
    """
    parameters:
        features: [batch_size, feature_dim]
        labels: [batch_size]
        eps: a small value to avoid log(0)
    """
    fea_sim = torch.matmul(features, features.t())
    soft_fea_sim = torch.softmax(fea_sim, dim=-1)
    soft_fea_sim = torch.clamp(soft_fea_sim, min=eps, max=1-eps)
    # labels相同的features的相似度之和
    mask = labels.unsqueeze(1) == labels.unsqueeze(0)
    loss = -torch.sum(torch.log(soft_fea_sim[mask]))
    return loss

# 3. Knowledge Distillation Loss
def KD_loss(teacher_logits: torch.tensor, student_logits: torch.tensor, T=1, eps=1e-12):
    """
    parameters:
        teacher_logits: [batch_size, num_classes]
        student_logits: [batch_size, num_classes]
        T: temperature
        eps: a small value to avoid log(0)
    """
    teacher_logits = F.softmax(teacher_logits/T, dim=-1)
    student_logits = F.softmax(student_logits/T, dim=-1)
    teacher_logits = torch.clamp(teacher_logits, min=eps, max=1-eps)
    student_logits = torch.clamp(student_logits, min=eps, max=1-eps)
    loss = -torch.mean(torch.sum(teacher_logits * torch.log(student_logits), dim=-1))
    return loss

    
