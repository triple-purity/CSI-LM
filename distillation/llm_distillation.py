import argparse
import os
from typing import Optional, List, Tuple
import numpy as np
from tqdm.auto import tqdm
from itertools import chain
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from dataset.datasets import CSI_Dataset

from models.LM_Base import build_LLM2Rec
from models.StuModels import CSINet, TimeModule

from dataset.data import get_csi_data, get_cross_domain_csi_data
from utils.train_util import InfoCE, KD_loss, feature_loss
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
    parser.add_argument('--cross_domain', default=None, type=str)
    parser.add_argument('--antenna_num', default=3, type=int, help='the number of antenna')
    parser.add_argument('--time_length', default=700, type=int)
    parser.add_argument('--extract_method', default='amplitude', type=str, help='amplitude or csi-ratio or dfs')
    parser.add_argument('--data_norm_type', default='min_max_1', type=str, help='min_max_1 or min_max_2 or mean_std')
    parser.add_argument('--data_key', default='csi_data', type=str)
    
    # create lm_model--teacher params
    parser.add_argument('--llm_name', default='openai-community/gpt2', type=str)
    parser.add_argument('--d_model', default=1024, type=int)
    parser.add_argument('--input_dim', default=90, type=int)
    parser.add_argument('--token_kernels', default=[5, 11, 21], type=int, nargs='+')
    parser.add_argument('--n_heads', default=8, type=int)
    parser.add_argument('--llm_layers', default=12, type=int)
    parser.add_argument('--start_layer', default=0, type=int)
    parser.add_argument('--frozen_llm_layer', default=12, type=int)
    parser.add_argument('--lora', default=False, type=bool)
    parser.add_argument('--add_embed_layer', default=False, type=bool)
    parser.add_argument('--reprogramming', default=False, type=bool)

    # create TimeModule--student params
    parser.add_argument('--action_num', default=6, type=int)
    parser.add_argument('--num_encoder', default=4, type=int)
    parser.add_argument('--decoder_mask', default=True, type=bool)
    parser.add_argument('--pos_learn', default=False, type=bool)

    #train model params
    parser.add_argument('--epoch', default=5, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--noise', default=0., type=float)
    parser.add_argument('--lr', default=4e-5, type=float)
    parser.add_argument('--scheduler', default=True, type=bool)
    parser.add_argument('--weight_deacy', default=0.0, type=float)
    parser.add_argument('--label_smooth_rate', default=0.0, type=float)
    parser.add_argument('--fea_gama', default=0.1, type=int)
    parser.add_argument('--contrastive_alpha', default=0.0001, type=float)
    parser.add_argument('--kd_beta', default=0.15, type=int)
    args = parser.parse_args()
    return args

def train_model(teacher_model: nn.Module, student_model: nn.Module, train_data: DataLoader, eval_data: DataLoader,
                start_epoch, epochs, optimizer, scheduler, device, args, eval=True):
    torch.autograd.set_detect_anomaly(True) 
    # 确定损失函数组成
    cls_loss = nn.CrossEntropyLoss(label_smoothing=args.label_smooth_rate)
    Avg_Loss, Acc= list(), list()

    for epoch in range(start_epoch, epochs):
        print('*** Start Training Model with Train Data ***')
        teacher_model.train()
        student_model.train()
        teacher_model.to(device)
        student_model.to(device)

        pred_labels = []
        targets = []
        avg_loss, avg_sup_loss_stu, avg_sup_loss_tea, avg_fea_loss, avg_con_loss, avg_kd_loss = 0, 0, 0, 0, 0, 0
        bar = tqdm(enumerate(train_data), total=len(train_data))
        for i,(inputs, action_labels, _) in bar:
            if args.extract_method == 'csi-ratio':
                B, K, T, C = inputs.shape
                inputs = inputs.reshape(B*K, T, -1)
                action_labels = action_labels.flatten(0)
            inputs = inputs.to(device)
            action_labels = action_labels.to(device)

            stu_out_dict = student_model(inputs, decoder_mask=args.decoder_mask, return_embed=True, return_feature=True)
            input_embeds, stu_features, stu_logits = stu_out_dict['embeds'], stu_out_dict['features'], stu_out_dict['logits']
            tea_out_dict = teacher_model(input_embeds, return_feature=True)
            tea_features, tea_logits =  tea_out_dict['features'], tea_out_dict['logits'] 
            
            sup_loss_stu = cls_loss(stu_logits, action_labels)
            sup_loss_tea = cls_loss(tea_logits, action_labels)
            fea_loss = feature_loss(stu_features, tea_features)
            con_loss = InfoCE(torch.mean(stu_features[-1], dim=1), action_labels)    # 对比损失，拉近相同标签的样本
            kd_loss = KD_loss(tea_logits, stu_logits)
            loss = (1-args.kd_beta)*sup_loss_stu + args.kd_beta * kd_loss + sup_loss_tea + args.fea_gama*fea_loss + args.contrastive_alpha * con_loss

            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()

            # 正确率计算
            pred_label = torch.argmax(stu_logits, dim=-1)
            pred_labels.append(pred_label.cpu())
            targets.append(action_labels.cpu())
            # 平均损失计算
            avg_loss = (avg_loss*i+loss.item())/(i+1)
            avg_sup_loss_stu = (avg_sup_loss_stu*i+sup_loss_stu.item())/(i+1)
            avg_sup_loss_tea = (avg_sup_loss_tea*i+sup_loss_tea.item())/(i+1)
            avg_fea_loss = (avg_fea_loss*i+fea_loss.item())/(i+1)
            avg_con_loss = (avg_con_loss*i+con_loss.item())/(i+1)
            avg_kd_loss = (avg_kd_loss*i+kd_loss.item())/(i+1)

            bar.set_description(
                desc=f"Epoch:{epoch+1}/{epochs}--Loss:{avg_loss:.4f} ||Supervised_Loss_Tea:{avg_sup_loss_tea:.4f} && Supervised_Loss_Stu:{avg_sup_loss_stu:.4f} && Feature_Loss:{avg_fea_loss:.4f} && Contrastive_Loss:{avg_con_loss:.4f} && KD_Loss:{avg_kd_loss:.4f}")
         
        preds = torch.cat(pred_labels).numpy()
        targets = torch.cat(targets).numpy()
        stu_train_acc = accuracy_score(targets, preds)
        print(f"Train Time the Accuracy Score of Model is:{stu_train_acc:.4f}")
        
        if args.scheduler:
            scheduler.step()

        if eval:
            print("*** Start Evaluation with Eval Data ***")
            eval_avg_loss, eval_acc = eval_model(student_model, eval_data, device, args=args)
        
        Avg_Loss.append((avg_sup_loss_stu, eval_avg_loss))
        Acc.append((stu_train_acc, eval_acc))
    return Avg_Loss, Acc

global_eval_acc = 0.80
def eval_model(model, eval_data, device, args):
    global global_eval_acc
    model.to(device)
    model.eval()
    loss_fun = nn.CrossEntropyLoss()

    pred_labels = []
    targets = []

    eval_avg_loss = 0
    with torch.no_grad():
        bar = tqdm(enumerate(eval_data), total=len(eval_data))
        for i,(inputs, action_labels, _) in bar:
            if args.extract_method == 'csi-ratio':
                B, K, T, C = inputs.shape
                inputs = inputs.reshape(B*K, T, -1)
                action_labels = action_labels.flatten(0)

                inputs = inputs.to(device)
                action_labels = action_labels.to(device)
            
                action_logits, _ = model.predict(inputs, decoder_mask=args.decoder_mask)
                eval_loss = loss_fun(action_logits, action_labels)
                eval_avg_loss = (eval_avg_loss * i + eval_loss.item())/(i+1)

                action_logits = F.softmax(action_logits, dim=-1)
                action_logits = torch.mean(action_logits.reshape(B, K, -1), dim=1)
                pre_actions = torch.argmax(action_logits, dim=-1)
                pred_labels.append(pre_actions.cpu())
                targets.append(action_labels[::K].cpu())
            else:
                inputs = inputs.to(device)
                action_labels = action_labels.to(device)
            
                action_logits, pre_actions = model.predict(inputs, decoder_mask=args.decoder_mask)
                eval_loss = loss_fun(action_logits, action_labels)
                eval_avg_loss = (eval_avg_loss * i + eval_loss.item())/(i+1)

                pred_labels.append(pre_actions.cpu())
                targets.append(action_labels.cpu())

        preds = torch.cat(pred_labels).numpy()
        targets = torch.cat(targets).numpy()
        eval_acc = accuracy_score(targets, preds)
        print(f"The Avg Loss of Model is:{eval_avg_loss}")
        print(f"Eval Time the Accuracy Score of Model is:{eval_acc:.4f}")
        if eval_acc > global_eval_acc:
            global_eval_acc = eval_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, f'student_model.pth'))
    return eval_avg_loss, eval_acc
def main():
    args = get_args_parser()
    print(args)
    # 1. load data
    print("***** Start Load Data *****")
    if args.cross_domain is None:
        print("*** Load Data not Cross Domain ***")
        train_datas, train_gesture_labels, train_domain_labels, eval_datas, eval_gesture_labels, eval_domain_labels, = get_csi_data(
            args.data_path,
            select_domains = args.data_domains,
        )
    else:
        print("*** Load Data Cross Domain ***")
        train_datas, train_gesture_labels, train_domain_labels, eval_datas, eval_gesture_labels, eval_domain_labels, = get_cross_domain_csi_data(
            args.data_path,
            select_domains = args.data_domains,
            cross_doamin = args.cross_domain,
        )

    select_data = 1
    train_dataset = CSI_Dataset(train_datas[select_data], train_gesture_labels[select_data], 
                                train_domain_labels[select_data], antenna_num=args.antenna_num, 
                                unified_length=args.time_length, 
                                extract_method=args.extract_method, 
                                data_key=args.data_key, norm_type=args.data_norm_type, noise=args.noise)
    eval_dataset = CSI_Dataset(eval_datas[select_data], eval_gesture_labels[select_data], 
                               eval_domain_labels[select_data], antenna_num=args.antenna_num,
                               unified_length=args.time_length, 
                               extract_method=args.extract_method, 
                               data_key=args.data_key, norm_type=args.data_norm_type)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    # 3. Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    teacher_model = build_LLM2Rec(
        class_num=args.action_num,
        llm_name=args.llm_name,
        d_model=args.d_model,
        input_dim=args.input_dim,
        token_kernels=args.token_kernels,
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
        token_kernels=args.token_kernels,
        llm_name=args.llm_name,
        d_model=args.d_model,
        embed_size=teacher_model.d_llm,
        n_heads=args.n_heads,
        num_encoder=args.num_encoder,
        pos_learn=args.pos_learn,
    )
    
    # 4. Optimizer
    model_optimizer = optim.Adam(
        chain(teacher_model.parameters(),student_model.parameters()),
        lr=args.lr, 
        weight_decay=args.weight_deacy
    )
   
    # 5. Scheduler
    scheduler = None
    if args.scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(model_optimizer, T_max= args.epoch, eta_min=5e-6)
        
    # 6. Train
    avg_loss, acc = train_model(teacher_model, student_model, train_data=train_loader, eval_data=eval_loader, 
                                start_epoch=0, epochs=args.epoch, optimizer=model_optimizer, scheduler=scheduler, 
                                device=device, args=args, eval=True)
    
    # 7. display loss and acc
    train_loss, eval_loss = [i[0] for i in avg_loss], [i[1] for i in avg_loss]
    train_acc, eval_acc = [i[0] for i in acc], [i[1] for i in acc]

    # 绘制损失曲线
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(eval_loss, label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(eval_acc, label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join('../../data/photos', 'llm_dis_stu_loss_acc.png'))
    plt.show()
if __name__ == '__main__':
    main()

