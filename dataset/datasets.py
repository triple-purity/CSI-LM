import os
from typing import Optional, Tuple, List

import scipy as sp
import scipy.io as sio
import numpy as np

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


def data_norm(x: torch.Tensor, norm_type: str = "min_max_1"):
    '''
    param x: torch.Tensor, shape (T, C)
    param norm_type: str, norm type, "min_max_1" or "min_max_2" or "mean_std"
    return normed_x: torch.Tensor, shape (T, C)
    '''
    if norm_type == "min_max_2":
        min_x = x.min()
        max_x = x.max()
        return (x-min_x)*2/(max_x-min_x)-1
    elif norm_type == "mean_std":
        mean_x = x.mean(dim=0, keepdim=True)
        std_x = x.std(dim=0, keepdim=True)
        return (x-mean_x)/std_x
    elif norm_type == "min_max_1":
        min_x = x.min()
        max_x = x.max()
        return (x-min_x)/(max_x-min_x)
    else:
        raise ValueError("Invalid norm type: {}".format(norm_type))

class CSI_Dataset(Dataset):
    """CSI dataset."""
    def __init__(self, 
                 data_path: str, 
                 data_names:List[str], 
                 labels: List[str], 
                 time_length: int, 
                 norm_type: str='min_max_1',
                ):
        """Initialize dataset.

        Args:
            data_path (str): path to the dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_path = data_path
        self.data_names = [os.path.join(data_path, name) for name in data_names]
        self.labels = labels
        self.time_length = time_length
        self.norm_type = norm_type
    def __len__(self):
        return len(self.data_names)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns the data and label at the given index.
        csi data 是复数矩阵，该如何修改:
        1. 仅保留幅度
        """
        cur_label = int(self.labels[idx])-1
        data_file = self.data_names[idx]
        try:
            csi_mat = sio.loadmat(data_file)
            data_key = list(csi_mat.keys())[-1]
        except FileNotFoundError:
            print("File not found: {}".format(data_file))
        
        # CSI Data Process
        # 目前仅保留幅度
        csi_mat = csi_mat[data_key]

        # transform to tensor
        tensor_csi = torch.tensor(csi_mat, dtype=torch.float32)
        tensor_csi = self.data_process(tensor_csi)
        tensor_csi = data_norm(tensor_csi, self.norm_type)
        return tensor_csi, cur_label

    def data_process(self, tensor_csi):
        time_len, _ = tensor_csi.shape
        if time_len > self.time_length:
            tensor_csi = tensor_csi[:self.time_length]
        elif time_len < self.time_length:
            copy_csi = tensor_csi[-1].unsqueeze(0)
            copy_csi = copy_csi.repeat(self.time_length-time_len, 1)
            tensor_csi = torch.cat([tensor_csi, copy_csi], dim=0)
        else:
            pass
        return tensor_csi


class HAR_Dataset(Dataset):
    def __init__(self, data_path:str, time_length:int=2000, norm_type: str='min_max_1'):
        self.time_length = time_length
        self.norm_type = norm_type

        folders = os.listdir(data_path)
        self.label2id = {label:i for i,label in enumerate(folders)}
        self.datas = list()
        for folder in folders:
            folder_path = os.path.join(data_path, folder)
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                self.datas.append((file_path, self.label2id[folder]))
    
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        data_path, label = self.datas[idx]
        try:
            csi_mat = sio.loadmat(data_path)
            data_key = list(csi_mat.keys())[-1]
        except FileNotFoundError:
            print("File not found: {}".format(data_path))
        
        csi_mat = csi_mat[data_key]

        # transform to tensor
        tensor_csi = torch.tensor(csi_mat, dtype=torch.float32)
        tensor_csi = tensor_csi.permute(1, 0)
        tensor_csi = self.data_process(tensor_csi)
        tensor_csi = data_norm(tensor_csi, self.norm_type)
        return tensor_csi, label

    def data_process(self, tensor_csi):
        time_len, _ = tensor_csi.shape
        if time_len > self.time_length:
            tensor_csi = tensor_csi[:self.time_length]
        elif time_len < self.time_length:
            copy_csi = tensor_csi[-1].unsqueeze(0)
            copy_csi = copy_csi.repeat(self.time_length-time_len, 1)
            tensor_csi = torch.cat([tensor_csi, copy_csi], dim=0)
        else:
            pass
        return tensor_csi
    

class BVP_Dataset(Dataset):
    def __init__(self, 
                 data_path: str, 
                 data_names:List[str], 
                 labels: List[str], 
                 time_length: int = 22, 
                 norm_type: str='min_max_1',
                ):
        """
        Initialize dataset.

        Args:
            data_path (str): path to the dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_path = data_path
        self.data_names = [os.path.join(data_path, name) for name in data_names]
        self.labels = labels
        self.time_length = time_length
        self.norm_type = norm_type
    def __len__(self):
        return len(self.data_names)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        cur_label = int(self.labels[idx])-1
        data_file = self.data_names[idx]
        try:
            bvp_mat = sio.loadmat(data_file)
            data_key = list(bvp_mat.keys())[-1]
        except FileNotFoundError:
            print("File not found: {}".format(data_file))
        
        bvp_mat = bvp_mat[data_key]

        # transform to tensor
        tensor_bvp = torch.tensor(bvp_mat, dtype=torch.float32)
        _, _, cur_len = tensor_bvp.shape
        tensor_bvp = tensor_bvp.permute(2,0,1).reshape(cur_len, -1)
        tensor_bvp = self.data_process(tensor_bvp)
        # normalize
        # tensor_bvp = (tensor_bvp - 0.0025)/0.0119
        return tensor_bvp, cur_label

    def data_process(self, tensor_bvp):
        time_len, dim = tensor_bvp.shape
        if time_len > self.time_length:
            tensor_bvp = tensor_bvp[:self.time_length]
        elif time_len < self.time_length:
            copy_bvp = torch.zeros(self.time_length-time_len, dim)
            tensor_bvp = torch.cat([tensor_bvp, copy_bvp], dim=0)
        else:
            pass
        return tensor_bvp
    

class DFS_Dataset(Dataset):
    def __init__(self, 
                 data_path: str, 
                 data_names:List[str], 
                 labels: List[str], 
                 mat_key:str = 'doppler_spectrum_r',
                 receiver_num: int = 1,
                 time_length: int = 2000, 
                 norm_type: str='min_max_1',
                ):
        
        self.data_path = data_path
        self.data_names = [os.path.join(data_path, name) for name in data_names]
        self.labels = labels
        self.mat_key = mat_key
        self.receiver_num = receiver_num
        self.time_length = time_length
        self.norm_type = norm_type

    def __len__(self): 
        return len(self.data_names)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        cur_label = int(self.labels[idx])-1
        data_file = self.data_names[idx]
        try:
            mat_file = sio.loadmat(data_file)
        except FileNotFoundError:
            print("File not found: {}".format(data_file))
        
        dfs_mat = mat_file[self.mat_key]
        dfs_re = dfs_mat[self.receiver_num,:,:]
        # transform to tensor
        tensor_dfs = torch.tensor(dfs_re, dtype=torch.float32)
        tensor_dfs = tensor_dfs.permute(1, 0)   # (freq_bin, time_len) -> (time_len, freq_bin)
        tensor_dfs = self.data_process(tensor_dfs)

        # normalize :
        tensor_dfs = data_norm(tensor_dfs, self.norm_type)
        return tensor_dfs, cur_label
    
    def data_process(self, tensor_bvp):
        time_len, dim = tensor_bvp.shape
        if time_len > self.time_length:
            tensor_bvp = tensor_bvp[:self.time_length]
        elif time_len < self.time_length:
            copy_bvp = torch.zeros(self.time_length-time_len, dim)
            tensor_bvp = torch.cat([tensor_bvp, copy_bvp], dim=0)
        else:
            pass
        return tensor_bvp