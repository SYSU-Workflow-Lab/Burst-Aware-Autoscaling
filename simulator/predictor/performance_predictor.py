from util.constdef import get_cpu_predictor, get_res_predictor, read_profilng_data
from resource_predictor import filter_data
import numpy as np
import pandas as pd
import logging
import random
class CpuPredictor:
    def __init__(self):
        self.predictor = get_cpu_predictor()
        self.origin_data = get_origin_data()
        self.data_per_workload = self.origin_data['per_workload'].values
        self.data_instance = self.origin_data['ins'].values
        self.data_cpu_utilization = self.origin_data['cpu'].values
        self.data_response_time = self.origin_data['res'].values
        self.allow_bias = 0.05
        self.min_bias = 0.1
        self.min_number = 5

    
    def predict(self, data, instance = 5, use_sampling=False):
        """
        use_sampling代表是否可以使用采样方法代替预测

        """
        per_workload = data.item()
        if len(data.shape) == 1:
            data = data.reshape(-1,1)

        if not use_sampling:
            # 不允许采样，则直接预测
            return self.predictor.predict(data)
        
        down_threshold = per_workload*(1-self.allow_bias)
        if down_threshold > per_workload - self.min_bias:
            down_threshold = per_workload - self.min_bias
        up_threshold = per_workload*(1+self.allow_bias)
        if up_threshold < per_workload + self.min_bias:
            up_threshold = per_workload + self.min_bias
        # 1. 找到对应位置的数据
        per_workload_range = np.logical_and(self.data_per_workload < up_threshold, self.data_per_workload > down_threshold)
        
        if np.sum(per_workload_range) < self.min_number:
            return self.predictor.predict(data)
        else:
            aim_data = self.data_cpu_utilization[per_workload_range]
            return np.random.choice(aim_data,1)
    
    def predict_once(self,data, instance = 5,use_sampling = False):
        if isinstance(data,np.ndarray):
            assert len(data) == 1
            return self.predict(data, instance = instance,use_sampling=use_sampling).item()
        else:
            return self.predict(np.array([data]), instance = instance,use_sampling=use_sampling).item()

class ResPredictor:
    def __init__(self):
        self.predictor = get_res_predictor()
        self.origin_data = get_origin_data()
        self.data_per_workload = self.origin_data['per_workload'].values
        self.data_instance = self.origin_data['ins'].values
        self.data_cpu_utilization = self.origin_data['cpu'].values
        self.data_response_time = self.origin_data['res'].values
        self.allow_bias = 0.05
        self.min_bias = 0.1
        self.min_number = 5
    
    def predict(self, data, instance=5,use_sampling=False):
        # NOTE SVR预测的偏差值过大，如果超过20的话，按20通算
        origin_instance = instance
        if instance > max(self.data_instance):
            instance = max(self.data_instance)
        # logging.info(f"{origin_instance} to {instance}")

        per_workload = data.item()
        if len(data.shape) == 1:
            data = data.reshape(-1,1)
        if not use_sampling:
            # 不允许采样，则直接预测
            # return self.predictor.predict(np.array([per_workload,instance]).reshape(1,2))
            return self.predictor.predict(data)

        # down_threshold = per_workload*(1-self.allow_bias)
        down_threshold = per_workload - 1
        if down_threshold > per_workload - self.min_bias:
            down_threshold = per_workload - self.min_bias
        # up_threshold = per_workload*(1+self.allow_bias)
        up_threshold = per_workload + 1
        if up_threshold < per_workload + self.min_bias:
            up_threshold = per_workload + self.min_bias

        per_workload_range = np.logical_and(self.data_per_workload < up_threshold, self.data_per_workload > down_threshold)
        if np.sum(per_workload_range) < self.min_number:
            # 样本不足
            return self.predictor.predict(data)
        else:
            aim_data = self.data_response_time[per_workload_range]
            return np.random.choice(aim_data,1)
    
    def predict_once(self,data, instance=5,use_sampling=False):
        if isinstance(data,np.ndarray):
            assert len(data) == 1
            return self.predict(data, instance=instance,use_sampling=use_sampling).item()
        else:
            return self.predict(np.array([data]), instance=instance,use_sampling=use_sampling).item()

def clear_by_3sigma(origin_data,aim):
    origin_data = origin_data.sort_values(by=["per_workload"])
    after_clear_list = []
    interval_num = 30
    ratio = 3
    for i in range(len(origin_data)//interval_num + 1):
        aim_pd = origin_data.iloc[i*interval_num:(i+1)*interval_num]
        aim_data = aim_pd[aim].values
        aim_mean = np.mean(aim_data)
        aim_std = np.std(aim_data)
        
        rest_data = aim_pd
        rest_data = rest_data[rest_data[aim]>aim_mean-ratio*aim_std]
        rest_data = rest_data[rest_data[aim]<aim_mean+ratio*aim_std]
        after_clear_list.append(rest_data)
    return pd.concat(after_clear_list)

def get_origin_data():
    """
       根据resource_predictor.py的逻辑，返回预测时所需要的原始数据
    """
    # file_list = ['2021_12_29.csv','2021_12_30.csv','2021_12_31.csv']
    file_list = ['26_08_01.csv']
    pd_data_list = []
    for file_name in file_list:
        pd_data_list.append(read_profilng_data(file_name=file_name))
    data = pd.concat(pd_data_list)
    # data2 = read_profilng_data(file_name="total.csv")
    # total_data = filter_data(data)
    total_data = data.dropna()
    # 需要清除掉error_ratio过高等的情况
    total_data = total_data[total_data['err'] < 1.0]
    total_data['per_workload'] = total_data['wkl'] / total_data['ins']
    total_data = total_data.sort_values(by=["per_workload"])
    # 使用函数按照3sigma原则进行清除
    # total_data = clear_by_3sigma(total_data,aim="res")
    # total_data = clear_by_3sigma(total_data,aim="cpu")
    return total_data

