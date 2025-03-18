import os
import numpy as np
from typing import List, Tuple, Optional

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit


domains = {'user':0, 'gesture':1, 'location':2, 'direction':3, 'repeat':4}
def name_split(file_name, split_domains: List[str]):
    file_name = file_name.split('.')[0]
    elements = file_name.split('-')[:-1]
    gesture_label = int(elements[1])
    select_elements = [elements[domains[d]] for d in split_domains]
    return select_elements, gesture_label

# get CSI Data
def get_csi_data(data_path: str, select_domains: List[str]):
    users = os.listdir(data_path)
    all_files, labels = [], []
    feas = []
    for user in users:
        user_path = os.path.join(data_path, user)
        files = os.listdir(user_path)
        for file_name in files:
            all_files.append(os.path.join(user_path, file_name))
            elements, label = name_split(file_name, select_domains)
            feas.append(elements)
            labels.append(label)

    all_files, feas ,labels= np.array(all_files), np.array(feas), np.array(labels)
    stra_split = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
    train_datas, train_labels, test_datas, test_labels = [], [], [], []
    for train_index, test_index in stra_split.split(X=all_files, y=feas):
        train_datas.append(all_files[train_index])
        train_labels.append(labels[train_index])
        test_datas.append(all_files[test_index])
        test_labels.append(labels[test_index]) 
    return train_datas, train_labels, test_datas, test_labels

# get SignFi Data
def get_signfi_data(data_path: str, test_size = 0.2):
    file_names = os.listdir(data_path)
    labels = np.array([int(file.split('.')[0].split('-')[-1]) for file in file_names])
    split_y= labels.reshape(-1, 1)

    stra_split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_datas, train_labels, test_datas, test_labels = None, None, None, None
    for train_index, test_index in stra_split.split(X=file_names, y=split_y):
        train_datas = file_names[train_index]
        train_labels = labels[train_index]
        test_datas = file_names[test_index]
        test_labels = labels[test_index]

    return train_datas, train_labels, test_datas, test_labels
