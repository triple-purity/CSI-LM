# 该文件用来存放模型训练相关函数
from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score

def train_model(model: nn.Module, train_data, start_epoch, epochs, optimizer, scheduler, 
                loss_fn, device, args, eval_data: Optional[DataLoader] = None, eval=False):
    Avg_Loss = list()

    for epoch in range(start_epoch, epochs):
        print('*** Start TRAINING Model ***')
        model.train()
        model.to(device)

        avg_loss = 0
        bar = tqdm(enumerate(train_data), total=len(train_data))
        for i,(inputs, labels) in bar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            pred_logits = model(inputs, args.reprogramming)
            
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
            
            Avg_Loss.append(avg_loss)

            bar.set_description(desc = f'Epoch {epoch}/{epochs}: model classification loss: {avg_loss:.4f}')
        if args.scheduler:
            scheduler.step()
        if eval:
            print("***** Start Evaluation with Eval Data *****")
            eval_model(model, eval_data, loss_fn, device, args=args)
        
        print(f"***** Now save model at epoch:{epoch} *****")
        model.cpu()
        # torch.save(model.state_dict(), os.path.join(args.output_dir, f'model_{epoch}.pth'))
    return Avg_Loss

def eval_model(model, eval_data, loss_fun, device, args):
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
            
            pre_logits, pre_labels = model.predict(inputs, args.reprogramming)
            eval_loss = loss_fun(pre_logits, labels)
            eval_avg_loss = (eval_avg_loss * i + eval_loss.item())/(i+1)

            pred_labels.append(pre_labels.cpu())
            targets.append(labels.cpu())

        preds = torch.cat(pred_labels).numpy()
        targets = torch.cat(targets).numpy()
        print(f"The Avg Loss of model is:{eval_avg_loss}")
        print(f"The Accuracy score of model is:{accuracy_score(targets, preds)}")