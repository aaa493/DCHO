import os
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random

def preprocess_data(data_dir, time_chunk_length, stride=2):
    all_X_time, all_A_time, all_H_time = [], [], []

    pt_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pt')])

    for pt_file in pt_files:
        file_path = os.path.join(data_dir, pt_file)
        dataset = torch.load(file_path, weights_only=True)

        X_time = dataset["X_time"]  # (T, N, D)
        A_time = dataset["A_time"]  # (T, N, N)
        H_time = dataset["H_time"]  # (T, N, N, N)

        # 单通道数据自动加维
        if X_time.ndim == 2:
            X_time = X_time.unsqueeze(-1)  # (T, N, 1)

        T = X_time.shape[0]
        for start in range(0, T - time_chunk_length + 1, stride):
            end = start + time_chunk_length
            all_X_time.append(X_time[start:end])
            all_A_time.append(A_time[start:end])
            all_H_time.append(H_time[start:end])

    # 拼接所有滑窗样本 (num_samples, T, N, *)
    all_X_time = torch.stack(all_X_time, dim=0)
    all_A_time = torch.stack(all_A_time, dim=0)
    all_H_time = torch.stack(all_H_time, dim=0)

    return {
        "X_time": all_X_time,
        "A_time": all_A_time,
        "H_time": all_H_time,
    }



class fmri_Dataset(Dataset):
    def __init__(self, data_dict):

        self.X_time = data_dict["X_time"] 
        self.A_time = data_dict["A_time"]   
        self.H_time = data_dict["H_time"]  

    def __len__(self):
        """返回数据集的大小"""
        return self.X_time.shape[0]

    def __getitem__(self, idx):
        """获取指定索引的数据"""
        X = self.X_time[idx]
        A = self.A_time[idx]
        H = self.H_time[idx]
        return X, A, H


def get_train_dataloader(data_dir, time_chunk_length, batch_size):
    print("processing train data...")
    combined_data_dict = preprocess_data(data_dir, time_chunk_length)
    fmri_dataset = fmri_Dataset(combined_data_dict)
    dataloader = DataLoader(fmri_dataset, batch_size = batch_size, shuffle=True)
    return dataloader


def get_test_dataloader(data_dir, time_chunk_length, batch_size):
    print("processing test data...")
    combined_data_dict = preprocess_data(data_dir, time_chunk_length)
    fmri_dataset = fmri_Dataset(combined_data_dict)
    dataloader = DataLoader(fmri_dataset, batch_size = batch_size, shuffle=True)
    return dataloader







