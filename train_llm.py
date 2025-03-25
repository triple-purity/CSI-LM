import argparse
import os
from typing import Optional, List, Tuple
import numpy as np
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from dataset.datasets import CSI_Dataset, DFS_Dataset
from model.LLM_Fintue import LLM2Rec, build_LLM2Rec
from model.GAN import CSI_GAN
from utils.train_util import DomainDeception, confidence_loss

from tqdm.auto import tqdm
from dataset.data import get_csi_data
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

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
    
    # create model params
    parser.add_argument('--action_num', default=2, type=int)
    parser.add_argument('--domain_num', default=4, type=int)
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
    parser.add_argument('--reprogramming', default=False, type=bool)

    #train model params
    parser.add_argument('--epoch', default=5, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--action_lr', default=2e-5, type=float)
    parser.add_argument('--domain_lr', default=1e-5, type=float)
    parser.add_argument('--action_scheduler', default=False, type=bool)
    parser.add_argument('--domain_scheduler', default=False, type=bool)
    parser.add_argument('--init_action_epoch', default=1, type=int)
    parser.add_argument('--init_domain_epoch', default=4, type=int)
    parser.add_argument('--weight_deacy', default=0.0, type=float)
    parser.add_argument('--label_smooth_rate', default=0.02, type=float)
    parser.add_argument('--domain_alpha', default=0.05, type=float)
    parser.add_argument('--conf_beta', default=0.05, type=int)
    args = parser.parse_args()
    return args

def train_model(model, train_data, start_epoch, epochs, optimizer: dict, scheduler: dict, 
                device, args, eval_data: Optional[DataLoader] = None, eval=False):
    cls_loss = nn.CrossEntropyLoss()
    Avg_Loss = list()

    torch.autograd.set_detect_anomaly(True) 
    
    for epoch in range(start_epoch, epochs):
        print('*** Start TRAINING Model ***')
        model.train()
        model.to(device)
        epoch_temp = epoch%(args.init_domain_epoch+args.init_action_epoch)

        pred_labels = []
        targets = []
        avg_action_loss, avg_domain_loss = 0, 0
        bar = tqdm(enumerate(train_data), total=len(train_data))
        for i,(inputs, action_labels, domain_labels) in bar:
            inputs = inputs.to(device)
            action_labels = action_labels.to(device)
            domain_labels = domain_labels.to(device)

            action_logits, domain_logits = model(inputs)
            
            if epoch_temp<args.init_domain_epoch:
                domain_loss = cls_loss(domain_logits, domain_labels)
                avg_domain_loss = (avg_domain_loss * (i) + domain_loss.item())/(i+1)
                domain_loss.backward()
                optimizer['domain'].step()
                optimizer['domain'].zero_grad()
            else:
                action_loss = cls_loss(action_logits, action_labels)
                domain_loss = cls_loss(domain_logits, domain_labels)
                conf_loss = confidence_loss(action_logits)
                avg_action_loss = (avg_action_loss * i + action_loss.item())/(i+1)
                loss = action_loss - args.domain_alpha * domain_loss + args.conf_beta*conf_loss
                loss.backward()
                optimizer['action'].step()
                optimizer['action'].zero_grad()

            bar.set_description(
                desc = f'Epoch {epoch}/{epochs}: Avg Action Loss: {avg_action_loss:.4f}|| Avg Domain Loss: {avg_domain_loss:.4f}'
            )

            #正确率
            pred_label = torch.argmax(action_logits, dim=-1)
            pred_labels.append(pred_label.cpu())
            targets.append(action_labels.cpu())
        
        Avg_Loss.append(avg_action_loss)
        preds = torch.cat(pred_labels).numpy()
        targets = torch.cat(targets).numpy()
        print(f"Train Time the Accuracy Score of Model is:{accuracy_score(targets, preds)}")

        if args.action_scheduler:
            if epoch_temp>=args.init_domain_epoch:
                scheduler['action'].step()
        if args.domain_scheduler:
            if epoch_temp<args.init_domain_epoch:
                scheduler['domain'].step()

        if eval and epoch_temp>=args.init_domain_epoch:
            print("***** Start Evaluation with Eval Data *****")
            eval_model(model, eval_data, device, args=args)
    return Avg_Loss

def eval_model(model, eval_data, device, args):
    model.to(device)
    model.eval()
    loss_fun = nn.CrossEntropyLoss()

    pred_labels = []
    targets = []

    print("*** Evaluation ***")
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
        print(f"The Avg Loss of model is:{eval_avg_loss}")
        print(f"The Accuracy score of model is:{accuracy_score(targets, preds)}")
        
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
    
    gan_model = CSI_GAN(
        action_num=args.action_num,
        domain_num=args.domain_num,
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
        reprogramming=args.reprogramming,
    )
    
    # 4. Optimizer
    action_optimizer = optim.Adam(
        chain(gan_model.feature_extracter.parameters(),gan_model.action_net.parameters()),
        lr=args.action_lr, 
        weight_decay=args.weight_deacy
    )
    domain_optimizer = optim.Adam(
        gan_model.domain_net.parameters(),
        lr=args.domain_lr, 
        weight_decay=args.weight_deacy
    )
    optimizer = {'action': action_optimizer, 'domain':domain_optimizer}
   
    # 5. Scheduler
    scheduler = {}
    if args.action_scheduler:
        action_scheduler = optim.lr_scheduler.CosineAnnealingLR(action_optimizer, T_max= args.epoch, eta_min=5e-6)
        scheduler['action'] = action_scheduler
    if args.domain_scheduler:
        domain_scheduler = optim.lr_scheduler.CosineAnnealingLR(domain_optimizer, T_max= args.epoch, eta_min=5e-6)
        scheduler['domain'] = domain_scheduler

    # 6. Train
    train_model(gan_model, train_loader, 0, args.epoch, optimizer, scheduler,
                device, args, eval_data=eval_loader, eval=True)
    
if __name__ == '__main__':
    main()