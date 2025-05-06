import os
import numpy as np
from typing import List, Tuple, Optional

from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit


domains = {'user':0, 'gesture':1, 'location':2, 'direction':3, 'repeat':4}
def name_split(file_name, split_domains: List[str], cross_doamin: str=None):
    file_name = file_name.split('.')[0]
    elements = file_name.split('-')[:5]
    gesture_label = int(elements[1])
    domain_label = (int(elements[2])-1)*5 + int(elements[3])-1
    select_elements = [elements[domains[d]] for d in split_domains]
    if cross_doamin is not None:
        return select_elements, gesture_label, domain_label, int(elements[domains[cross_doamin]])
    return select_elements, gesture_label, domain_label

# get CSI Data
def get_csi_data(data_path: str, select_domains: List[str]):
    users = os.listdir(data_path)
    all_files, gesture_labels, domain_labels = [], [], []
    feas = []
    for user in users:
        user_path = os.path.join(data_path, user)
        files = os.listdir(user_path)
        for file_name in files:
            all_files.append(os.path.join(user_path, file_name))
            elements, gesture_label, domain_label = name_split(file_name, select_domains)
            feas.append(elements)
            gesture_labels.append(gesture_label)
            domain_labels.append(domain_label)

    all_files, feas= np.array(all_files), np.array(feas) 
    gesture_labels, domain_labels = np.array(gesture_labels), np.array(domain_labels)
    stra_split = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
    train_datas, train_gesture_labels, train_domain_labels = [], [], []
    test_datas, test_gesture_labels, test_domain_labels = [], [], []
    for train_index, test_index in stra_split.split(X=all_files, y=feas):
        train_datas.append(all_files[train_index])
        train_gesture_labels.append(gesture_labels[train_index])
        train_domain_labels.append(domain_labels[train_index])
        test_datas.append(all_files[test_index])
        test_gesture_labels.append(gesture_labels[test_index])
        test_domain_labels.append(domain_labels[test_index])
    return train_datas, train_gesture_labels, train_domain_labels, test_datas, test_gesture_labels, test_domain_labels


# get Cross_Domain CSI Data
def get_cross_domain_csi_data(data_path: str, select_domains: List[str], cross_doamin: str, cro_dom_num: int=5):
    users = os.listdir(data_path)
    all_files, gesture_labels, domain_labels = [[] for _ in range(cro_dom_num)], [[] for _ in range(cro_dom_num)], [[] for _ in range(cro_dom_num)]
    for user in users:
        user_path = os.path.join(data_path, user)
        files = os.listdir(user_path)
        for file_name in files:
            _, gesture_label, domain_label, cross_doamin_label = name_split(file_name, select_domains, cross_doamin)
            all_files[cross_doamin_label-1].append(os.path.join(user_path, file_name))
            gesture_labels[cross_doamin_label-1].append(gesture_label)
            domain_labels[cross_doamin_label-1].append(domain_label)
        
    # 构建训练集和测试集
    train_datas, train_gesture_labels, train_domain_labels = [], [], []
    test_datas, test_gesture_labels, test_domain_labels = [], [], []

    for i in range(cro_dom_num):
        # 将第i个域作为测试集
        test_data = np.array(all_files[i])
        test_gesture_label = np.array(gesture_labels[i])
        test_domain_label = np.array(domain_labels[i])

        # 其余域合并为训练集
        train_data = np.concatenate([np.array(all_files[j]) for j in range(cro_dom_num) if j != i], axis=0)
        train_gesture_label = np.concatenate([np.array(gesture_labels[j]) for j in range(cro_dom_num) if j != i], axis=0)
        train_domain_label = np.concatenate([np.array(domain_labels[j]) for j in range(cro_dom_num) if j != i], axis=0)

        train_datas.append(train_data)
        train_gesture_labels.append(train_gesture_label)
        train_domain_labels.append(train_domain_label)
        test_datas.append(test_data)
        test_gesture_labels.append(test_gesture_label)
        test_domain_labels.append(test_domain_label)

    return train_datas, train_gesture_labels, train_domain_labels, test_datas, test_gesture_labels, test_domain_labels
    

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

