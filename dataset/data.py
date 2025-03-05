import os
import numpy as np
from typing import List, Tuple, Optional

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

def split_file_name(file_name: str,):
    domains = ['users', 'gesture_type', 'tosor_loc', 'face_ori', 'rec_dev']
    atris = file_name.split('.')[0].split('-')
    atris.pop(-2)
    atris_dict = { d:a for d, a in zip(domains, atris)}
    return atris_dict

# Split CSI data with Specific Domain
def get_csi_data(data_path: str, req_domains: List[str], rex=2,test_size = 0.2):
    domains = ['users', 'gesture_type', 'tosor_loc', 'face_ori', 'rec_dev']
    if 'gesture_type' not in req_domains:
        raise ValueError("The 'gesture_type' must be in the req_domains.")
    # assert the value in req_domainss must be in the domains
    for item in req_domains:
        if item not in domains:
            raise ValueError(f"{item} is not the standard value.")

    # file_name user--gesture_type--torso_location--face_orientation--repeat_num--receive_device
    file_names = os.listdir(data_path)
    file_names = filter(lambda x: x.split('.')[0].split('-')[-1]==f'r{rex}', file_names)
    file_names = np.array(list(file_names))

    domains_res = {d:[] for d in req_domains}
    for name in file_names:
        atris_dict = split_file_name(name)
        for d in req_domains:
            domains_res[d].append(atris_dict[d])
    
    split_y = np.array(list(domains_res.values())).T
    # split file_names with domains_res
    # split(X,y), the shape of y is (n_samples, n_labels)
    stra_split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_datas, train_labels, test_datas, test_labels = None, None, None, None
    for train_index, test_index in stra_split.split(X=file_names, y=split_y):
        train_datas = file_names[train_index]
        train_labels = np.array(domains_res['gesture_type'])[train_index]
        test_datas = file_names[test_index]
        test_labels = np.array(domains_res['gesture_type'])[test_index]

    return train_datas, train_labels, test_datas, test_labels

# Split CSI data with Specific Domain
def get_dfs_data(data_path: str, req_domains: List[str],test_size = 0.2):
    domains = ['users', 'gesture_type', 'tosor_loc', 'face_ori']
    if 'gesture_type' not in req_domains:
        raise ValueError("The 'gesture_type' must be in the req_domains.")
    # assert the value in req_domainss must be in the domains
    for item in req_domains:
        if item not in domains:
            raise ValueError(f"{item} is not the standard value.")

    # file_name user--gesture_type--torso_location--face_orientation--repeat_num--receive_device
    file_names = os.listdir(data_path)
    file_names = np.array(file_names)

    domains_res = {d:[] for d in req_domains}
    for name in file_names:
        atris_dict = split_file_name(name)
        for d in req_domains:
            domains_res[d].append(atris_dict[d])
    
    split_y = np.array(list(domains_res.values())).T
    # split file_names with domains_res
    # split(X,y), the shape of y is (n_samples, n_labels)
    stra_split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_datas, train_labels, test_datas, test_labels = None, None, None, None
    for train_index, test_index in stra_split.split(X=file_names, y=split_y):
        train_datas = file_names[train_index]
        train_labels = np.array(domains_res['gesture_type'])[train_index]
        test_datas = file_names[test_index]
        test_labels = np.array(domains_res['gesture_type'])[test_index]

    return train_datas, train_labels, test_datas, test_labels