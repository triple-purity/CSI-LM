import argparse
import os
from typing import Optional, List, Tuple
import numpy as np
from tqdm.auto import tqdm
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from dataset.datasets import CSI_Dataset, DFS_Dataset

from models.embed import PositionalEmbedding, TokenEmbedding
from models.LM_Base import build_LLM2Rec
from models.StuModels import CSINet, TimeModule

from dataset.data import get_csi_data
from utils.train_util import InfoCE, KD_loss
from sklearn.metrics import accuracy_score, precision_score

def get_args_parser():
    parser = argparse.ArgumentParser()
    # data path
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--data_domains', 
                        default=['user', 'gesture', 'location', 'direction'], 
                        nargs='+', type=str,
                        help='the value must be in the [user, gesture, location, direction]')

    # data process params
    parser.add_argument('--antenna_num', default=3, type=int, help='the number of antenna')
    parser.add_argument('--time_length', default=500, type=int)
    parser.add_argument('--extract_method', default='amplitude', type=str, help='amplitude or csi-ratio or dfs')
    parser.add_argument('--data_norm_type', default='min_max_1', type=str, help='min_max_1 or min_max_2 or mean_std')
    parser.add_argument('--data_key', default='csi_data', type=str)
    
    # create lm_model--teacher params
    parser.add_argument('--llm_name', default='unsloth/Qwen2.5-1.5B', type=str)
    parser.add_argument('--d_model', default=1024, type=int)
    parser.add_argument('--input_dim', default=90, type=int)
    parser.add_argument('--token_kernels', default=[5, 11, 21], type=int, nargs='+')
    parser.add_argument('--trans_layer', default=4, type=int)
    parser.add_argument('--n_heads', default=8, type=int)
    parser.add_argument('--llm_layers', default=12, type=int)
    parser.add_argument('--start_layer', default=0, type=int)
    parser.add_argument('--frozen_llm_layer', default=12, type=int)
    parser.add_argument('--lora', default=False, type=bool)
    parser.add_argument('--add_embed_layer', default=False, type=bool)
    parser.add_argument('--reprogramming', default=False, type=bool)

    # create TimeModule--student params
    parser.add_argument('--action_num', default=2, type=int)
    parser.add_argument('--num_encoder', default=4, type=int)

    #train model params
    parser.add_argument('--epoch', default=5, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--lr', default=2e-5, type=float)
    parser.add_argument('--scheduler', default=False, type=bool)
    parser.add_argument('--weight_deacy', default=0.0, type=float)
    parser.add_argument('--label_smooth_rate', default=0.05, type=float)
    parser.add_argument('--contrastive_alpha', default=1., type=float)
    parser.add_argument('--feature_beta', default=0.01, type=int)
    args = parser.parse_args()
    return args

def train_model(teacher_model: nn.Module, student_model: nn.Module, train_data: DataLoader, eval_data: DataLoader,
                start_epoch, epochs, optimizer, scheduler, device, args):
    torch.autograd.set_detect_anomaly(True) 
    # 确定损失函数组成
    cls_loss = nn.CrossEntropyLoss(args.label_smooth_rate)
    Avg_Loss = list()

    for epoch in range(start_epoch, epochs):
        print('*** Start Training Model with Train Data ***')
        teacher_model.train()
        student_model.train()
        teacher_model.to(device)
        student_model.to(device)

        pred_labels = []
        targets = []
        avg_loss, avg_sup_loss, avg_con_loss, avg_kd_loss = 0, 0, 0, 0
        bar = tqdm(enumerate(train_data), total=len(train_data))
        for i,(inputs, action_labels, _) in bar:
            inputs = inputs.to(device)
            action_labels = action_labels.to(device)

            return_dict = student_model(inputs, return_embed=True, return_feature=True)
            input_embeds, stu_features, action_logits = return_dict['embeds'], return_dict['features'], return_dict['logits']
            tea_features = teacher_model(input_embeds)
            
            sup_loss = cls_loss(action_logits, action_labels)
            con_loss = InfoCE(stu_features, action_labels)    # 对比损失，拉近相同标签的样本
            kd_loss = KD_loss(tea_features, stu_features)
            loss = sup_loss + args.contrastive_alpha * con_loss + args.feature_beta * kd_loss 

            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()

            # 正确率计算
            pred_label = torch.argmax(action_logits, dim=-1)
            pred_labels.append(pred_label.cpu())
            targets.append(action_labels.cpu())
            # 平均损失计算
            avg_loss = (avg_loss*i+loss.item())/(i+1)
            avg_sup_loss = (avg_sup_loss*i+sup_loss.item())/(i+1)
            avg_con_loss = (avg_con_loss*i+con_loss.item())/(i+1)
            avg_kd_loss = (avg_kd_loss*i+kd_loss.item())/(i+1)

            bar.set_description(
                desc=f"Epoch:{epoch+1}/{epochs}--Loss:{avg_loss:.4f} ||\
                        Supervised_Loss:{avg_sup_loss:.4f} && Contrastive_Loss:{avg_con_loss.item():.4f} && \
                        KD_Loss:{avg_kd_loss.item():.4f}")
        
        Avg_Loss.append(avg_loss)
        preds = torch.cat(pred_labels).numpy()
        targets = torch.cat(targets).numpy()
        print(f"Train Time the Accuracy Score of Model is:{accuracy_score(targets, preds):.4f}")
        
        if args.scheduler:
            scheduler.step()

        if eval:
            print("*** Start Evaluation with Eval Data ***")
            eval_model(student_model, eval_data, device, args=args)
    return Avg_Loss


def eval_model(model, eval_data, device, args):
    model.to(device)
    model.eval()
    loss_fun = nn.CrossEntropyLoss()

    pred_labels = []
    targets = []

    eval_avg_loss = 0
    with torch.no_grad():
        bar = tqdm(enumerate(eval_data), total=len(eval_data))
        for i,(inputs, action_labels, _) in bar:
            inputs = inputs.to(device)
            action_labels = action_labels.to(device)
            
            action_logits, pre_actions = model.predict(inputs)
            eval_loss = loss_fun(action_logits, action_labels)
            eval_avg_loss = (eval_avg_loss * i + eval_loss.item())/(i+1)

            pred_labels.append(pre_actions.cpu())
            targets.append(action_labels.cpu())

        preds = torch.cat(pred_labels).numpy()
        targets = torch.cat(targets).numpy()
        print(f"The Avg Loss of Model is:{eval_avg_loss}")
        print(f"Eval Time the Accuracy Score of Model is:{accuracy_score(targets, preds)}")


def main():
    args = get_args_parser()
    print(args)
    # 1. load data
    print("***** Start Load Data *****")
    train_datas, train_gesture_labels, train_domain_labels, eval_datas, eval_gesture_labels, eval_domain_labels, = get_csi_data(
        args.data_path,
        select_domains = args.data_domains,
    )

    train_dataset = CSI_Dataset(train_datas[2], train_gesture_labels[2], 
                                train_domain_labels[2], antenna_num=args.antenna_num, 
                                unified_length=args.time_length, 
                                extract_method=args.extract_method, 
                                data_key=args.data_key, norm_type=args.data_norm_type)
    eval_dataset = CSI_Dataset(eval_datas[2], eval_gesture_labels[2], 
                               eval_domain_labels[2], antenna_num=args.antenna_num,
                               unified_length=args.time_length, 
                               extract_method=args.extract_method, 
                               data_key=args.data_key, norm_type=args.data_norm_type)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    # 3. Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    teacher_model = build_LLM2Rec(
        llm_name=args.llm_name,
        d_model=args.d_model,
        input_dim=args.input_dim,
        token_kernels=args.token_kernels,
        trans_layer=args.trans_layer,
        n_heads=args.n_heads,
        llm_layers=args.llm_layers,
        start_layer=args.start_layer,
        frozen_llm_layer=args.frozen_llm_layer,
        batch_seq_len=args.time_length,
        lora=args.lora,
        add_embed_layer=args.add_embed_layer,
        reprogramming=args.reprogramming,
    )

    # create student model
    student_model = TimeModule(
        class_num=args.action_num,
        input_dim=args.input_dim,
        embed_size=teacher_model.d_llm,
        n_heads=args.n_heads,
        num_encoder=args.num_encoder,
    )
    
    # 4. Optimizer
    model_optimizer = optim.Adam(
        chain(teacher_model.parameters(),student_model.parameters()),
        lr=args.action_lr, 
        weight_decay=args.weight_deacy
    )
   
    # 5. Scheduler
    scheduler = None
    if args.action_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(model_optimizer, T_max= args.epoch, eta_min=5e-6)

    # 6. Train
    train_model(teacher_model, student_model, train_loader, 0, args.epoch, model_optimizer, scheduler,
                device, args, eval_data=eval_loader, eval=True)
    
if __name__ == '__main__':
    main()

