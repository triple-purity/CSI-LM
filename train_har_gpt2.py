import argparse
import os
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset.datasets import HAR_Dataset
from model.GPT2_Base import build_GPT2FCLS

from tqdm.auto import tqdm
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

def get_args_parser():
    parser = argparse.ArgumentParser()
    # data path
    parser.add_argument('--train_data_path', type=str)
    parser.add_argument('--eval_data_path', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--data_type', default='typical', type=str)

    # data process params
    parser.add_argument('--split_rate', default=0.2, type=float)
    parser.add_argument('--time_length', default=2000, type=int)
    parser.add_argument('--data_norm_type', default='min_max_1', type=str, help='min_max_1 or min_max_2 or mean_std')

    # create model params
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--token_kernel', default=3, type=int)
    parser.add_argument('--reduce_ratio', default=1, type=int)
    parser.add_argument('--wave_num', default=114, type=int)
    parser.add_argument('--r_acnt', default=3, type=int)
    parser.add_argument('--gpt2_trans_layers', default=12, type=int)
    parser.add_argument('--frozen_gpt2_layer', default=8, type=int)

    #train model params
    parser.add_argument('--epoch', default=5, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--lr', default=2e-5, type=float)
    parser.add_argument('--label_smooth_rate', default=0.01, type=float)
    parser.add_argument('--accumulate_backword', default=0, type=int)
    args = parser.parse_args()
    return args

def train_model(model, train_data, start_epoch, epochs, optimizer, scheduler, 
                loss_fn, device, args, eval_data: Optional[DataLoader] = None, eval=False):
    Avg_Loss = dict()

    for epoch in range(start_epoch, epochs):
        print('*** Start TRAINING Model ***')
        model.train()
        model.to(device)

        avg_loss = 0
        bar = tqdm(enumerate(train_data), total=len(train_data))
        for i,(inputs, labels) in bar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            pred_logits = model(inputs)
            
            loss = loss_fn(pred_logits, labels)

            loss.backward()
            if args.accumulate_backword > 0:
                if (i+1)%args.accumulate_backword == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                optimizer.step()
                optimizer.zero_grad()

            avg_loss = (avg_loss * i + loss.item())/(i+1)
            if (i+1)%4 == 0:
                Avg_Loss[(i+1)*args.batch_size] = avg_loss
            bar.set_description(desc = f'Epoch {epoch}/{epochs}: model classification loss: {avg_loss:.4f}')
        scheduler.step()
        if eval:
            print("***** Start Evaluation with Eval Data *****")
            eval_model(model, eval_data, loss_fn, device)
        
        print(f"***** Now save model at epoch:{epoch} *****")
        model.cpu()
        torch.save(model.state_dict(), os.path.join(args.output_dir, f'model_{epoch}.pth'))
    return Avg_Loss

def eval_model(model, eval_data, loss_fun, device):
    model.to(device)
    model.eval()

    pred_labels = []
    targets = []

    print("*** Evaluation ***")
    eval_avg_loss = 0
    with torch.no_grad():
        bar = tqdm(enumerate(eval_data), total=len(eval_data))
        for i,(inputs, labels) in bar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            pre_logits, pre_labels = model.predict(inputs)
            eval_loss = loss_fun(pre_logits, labels)
            eval_avg_loss = (eval_avg_loss * i + eval_loss.item())/(i+1)

            pred_labels.append(pre_labels.cpu())
            targets.append(labels.cpu())

        preds = torch.cat(pred_labels).numpy()
        targets = torch.cat(targets).numpy()
        print(f"The Avg Loss of model is:{eval_avg_loss}")
        print(f"The Accuracy score of model is:{accuracy_score(targets, preds)}")
def main():
    args = get_args_parser()
    print(args)
    # 1. load data
    train_csi = HAR_Dataset(args.train_data_path, args.time_length, norm_type=args.data_norm_type)
    eval_csi = HAR_Dataset(args.eval_data_path, args.time_length, norm_type=args.data_norm_type)
    train_loader = DataLoader(train_csi, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_csi, batch_size=args.batch_size, shuffle=False)
    print("Build the DataLoader Successfully!!!")

    # 2. Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = build_GPT2FCLS(
        num_classes=args.num_classes, 
        token_kernel=args.token_kernel,
        reduce_ratio= args.reduce_ratio,
        gpt_trans_layer=args.gpt2_trans_layers,
        wave_num = args.wave_num,
        r_acnt = args.r_acnt, 
        frozen_gpt2_layer=args.frozen_gpt2_layer
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smooth_rate)

    # 3. Train
    train_model(model, train_loader, 0, args.epoch, optimizer, scheduler,
                loss_fn, device, args, eval_data=eval_loader, eval=True)
    
if __name__ == '__main__':
    main()