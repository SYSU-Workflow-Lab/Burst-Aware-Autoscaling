# 尝试使用神经网络进行预测
import torch
import torch.nn as nn
from torch.utils.data import Dataset
# 使用多层ANN进行预测
class PerformancePredictor(nn.Module):
    def __init__(self, enc_number = 1, hidden_num = 48, n_output = 1):
        super(PerformancePredictor, self).__init__()
        self.forward_net = nn.Sequential(
            nn.Linear(enc_number, hidden_num),
            nn.ELU(),
            nn.Linear(hidden_num, hidden_num),
            nn.ELU(),
            nn.Linear(hidden_num, hidden_num),
            nn.ELU(),
            nn.Linear(hidden_num, n_output),
        )

    def forward(self, x):
        x1 = self.forward_net(x)
        return x1

import numpy as np
from sklearn.model_selection import KFold
class CustomDataset(Dataset):
    def __init__(self, data_dict,aim="cpu_utilization"):
        super(CustomDataset, self).__init__()
        self.per_workload = data_dict['workload'] / data_dict['instance']
        self.instance = data_dict['instance']
        self.error = data_dict['error_ratio']
        self.per_workload = self.per_workload[self.error<1.0]
        kf = KFold(n_splits=10,shuffle=True)

        origin_target = data_dict[aim]
        origin_target = origin_target[self.error<1.0]
        train_data,test_data = next(kf.split(origin_target))
        self.train_data = train_data
        self.test_data = test_data
        origin_target = origin_target[train_data]
        self.per_workload = self.per_workload[train_data]

        self.x_mean = np.mean(self.per_workload)
        self.x_std = np.std(self.per_workload)
        self.y_mean = np.mean(origin_target)
        self.y_std = np.std(origin_target)
        self.per_workload = (self.per_workload - self.x_mean) / self.x_std
        origin_target = (origin_target - self.y_mean) / self.y_std
        self.target = origin_target

    def __getitem__(self, index):
        return self.per_workload[index], self.target[index]
        # return self.workload[index], self.target[index]

    def __len__(self):
        return len(self.train_data)
