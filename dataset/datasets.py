import os
from typing import Optional, Tuple, List

import scipy as sp
import scipy.io as sio
import numpy as np
import random

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from utils.preprocess_tools import calculate_csi_ratio, hampel_filter, phase_calibration, DWT_Denoise, pca_denoise, get_doppler_spectrum, resample_csi_sequence

random.seed(42)
def data_norm(x: torch.Tensor, norm_type: str = "min_max_1"):
    '''
    param x: torch.Tensor, shape (T, C)
    param norm_type: str, norm type, "min_max_1" or "min_max_2" or "mean_std"
    return normed_x: torch.Tensor, shape (T, C)
    '''
    if norm_type == "min_max_2":
        min_x = x.min(dim=-1, keepdim=True).values
        max_x = x.max(dim=-1, keepdim=True).values
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

def enhance_csi_ratio(csi_ratio):
    new_csi_ratio = csi_ratio.clone()
    s = random.randint(0,100)
    if s%3 == 0:
        new_csi_ratio = -new_csi_ratio
    elif s%3 == 1:
        new_csi_ratio = torch.flip(new_csi_ratio, dims=[1])
    else:
        new_csi_ratio = -torch.flip(new_csi_ratio, dims=[1])
    return new_csi_ratio

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
            origin_csi = sio.loadmat(data_path)
            data_key = list(origin_csi.keys())[-1]
        except FileNotFoundError:
            print("File not found: {}".format(data_path))
        
        origin_csi = origin_csi[data_key]

        # transform to tensor
        tensor_csi = torch.tensor(origin_csi, dtype=torch.float32)
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


class CSI_Dataset(Dataset):
    """CSI dataset."""
    def __init__(self,  
                 data_names:List[str], 
                 action_labels: List[int],
                 domain_labels: List[int], 
                 antenna_num: int,
                 unified_length: int = 500,
                 extract_method: str = 'amplitude',
                 data_key: str = 'csi_data', 
                 norm_type: str='min_max_1',
                 noise: float = 0.0,
                 train: bool = True,
                ):
        """
        Initialize dataset.

        Params:
            data_path (str): path to the dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
            extract_method (str): method to extract information from CSI data, must be in in ['amplitude', 'csi-ratio', 'dfs']
            data_key (str): key to extract CSI data from the mat file, default is 'csi_data', other in ['csi-ratio', 'dfs']
        """
        assert extract_method in ['amplitude', 'csi-ratio', 'dfs'], "Invalid extract method: {}".format(extract_method)

        self.data_files = data_names
        self.action_labels = action_labels
        self.domain_labels = domain_labels
        self.antenna_num = antenna_num
        self.unified_length = unified_length
        self.extract_method = extract_method
        self.data_key = data_key
        self.norm_type = norm_type
        self.noise_std = noise
        self.train = train
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        '''
        Returns the data and label at the given index.
        '''
        gestur_label = int(self.action_labels[idx])-1
        domain_label = int(self.domain_labels[idx])
        data_file = self.data_files[idx]
        try:
            origin_data = sio.loadmat(data_file)
            origin_data = origin_data[self.data_key]
            if len(origin_data.shape) ==2:
                if self.extract_method != 'dfs':
                    origin_data = origin_data.reshape(origin_data.shape[0], self.antenna_num, -1)
                """
                else:
                    origin_data = np.transpose(origin_data, axes=(1, 0))
                    origin_data = origin_data.reshape(origin_data.shape[0], self.antenna_num, -1)
                """
        except FileNotFoundError:
            print("File not found: {}".format(data_file))
        
        # Extrac Information from CSI Data
        if self.extract_method == 'amplitude':
            abs_csi = np.abs(origin_data)
            # denoised_csi = DWT_Denoise(abs_csi, level=5, wavelet='db3')
            denoised_csi = pca_denoise(abs_csi)
            resample_csi = resample_csi_sequence(denoised_csi, target_length=self.unified_length)
            resample_csi = resample_csi.reshape(resample_csi.shape[0], -1)
            tensor_csi = torch.tensor(resample_csi, dtype=torch.float32)
            tensor_csi = tensor_csi + torch.randn_like(tensor_csi) * self.noise_std
            tensor_csi = data_norm(tensor_csi, self.norm_type)
            return tensor_csi, gestur_label, domain_label
        elif self.extract_method == 'csi-ratio':
            """
            csi_ratio, antenna_index = calculate_csi_ratio(origin_csi)
            csi_ratio = np.concatenate((csi_ratio[:, :antenna_index, :], csi_ratio[:, antenna_index+1:, :]), axis=1)
            angle_csi_ratio = np.angle(csi_ratio)
            angle_csi_ratio = phase_calibration(hampel_filter(angle_csi_ratio))"
            """
            csi_ratio = origin_data
            resample_csi_ratio = resample_csi_sequence(csi_ratio, target_length=self.unified_length)
            resample_csi_ratio = resample_csi_ratio.reshape(resample_csi_ratio.shape[0], -1)
            tensor_csi_ratio = torch.tensor(resample_csi_ratio, dtype=torch.float32)
            tensor_csi_ratio = data_norm(tensor_csi_ratio, self.norm_type)
            
            tensor_csi_ratio = tensor_csi_ratio.reshape(self.unified_length, 2, -1)
            tensor_csi_ratio = tensor_csi_ratio.permute(1, 0, 2)
            if self.train:
                enh_csi_ratio = enhance_csi_ratio(tensor_csi_ratio)
                tensor_csi_ratio_all = torch.cat([enh_csi_ratio, tensor_csi_ratio], dim=0)
                gestur_label = torch.tensor([gestur_label]*4, dtype=torch.long)
                domain_label = torch.tensor([domain_label]*4, dtype=torch.long)
            else:
                tensor_csi_ratio_all = tensor_csi_ratio
                gestur_label = torch.tensor([gestur_label]*2, dtype=torch.long)
                domain_label = torch.tensor([domain_label]*2, dtype=torch.long)
            return tensor_csi_ratio_all, gestur_label, domain_label
        else:
            # extract doppler spectrum from csi data
            # resample_csi = resample_csi_sequence(origin_csi, target_length=self.unified_length)
            # doppler_spectrum, *_ = get_doppler_spectrum(origin_csi)
            doppler_spectrum = origin_data
            
            _, sample_index = doppler_spectrum.shape
            if sample_index >= self.unified_length:
                doppler_spectrum = doppler_spectrum[:, :self.unified_length]
            else:
                doppler_spectrum = np.concatenate([doppler_spectrum, np.zeros((doppler_spectrum.shape[0], self.unified_length-sample_index))], axis=1)
            '''
            resample_doppler_spectrum = resample_csi_sequence(doppler_spectrum, target_length=self.unified_length)
            resample_doppler_spectrum = resample_doppler_spectrum.reshape(resample_doppler_spectrum.shape[0], -1)
            '''
            tensor_dfs = torch.tensor(doppler_spectrum, dtype=torch.float32)
            tensor_dfs = tensor_dfs.permute(1, 0)
            tensor_dfs = tensor_dfs + torch.randn_like(tensor_dfs)*self.noise_std
            return tensor_dfs, gestur_label, domain_label
