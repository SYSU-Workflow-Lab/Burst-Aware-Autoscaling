# 本文件负责各种常量的定义和读取
import os
from joblib import load
import numpy as np
import pandas as pd
from pathlib import Path

import torch

LONGTERM_CONF_RATIO = 0.99 # 长期预测使用99%的置信区间
TOTAL_STEP = int(8e5)
# 环境用模块
TRAIN_MODE = 0
TEST_MODE = 1
VALI_MODE = 2

IF_COVER_BURST = True # 默认为False，如果开启了，则在burst部分允许使用Non-burst调度器的结果
IF_RETRAIN = False # 各种加了缓存的位置是否要开启retrain，建议在新训练的时候统一开启，之后验证等的时候再关闭
MAX_WORKLOAD_FACTOR = 8 # 目前暂定最大限制到4000
LONGTERM_START_POINT = 9540 # 目前来说应该设置为之前的180天+23个小时
MAX_WKL = 100
MAX_CPU = 100
MAX_RES = 50
MAX_INS = 100 # 修改原因：原来的91是按照最大保障来计算的，结果是一定会有部分实例发生违约
ACTION_MAX = 10

# 超参数
class ConfigClass:
    reward_bias_ratio = 0.5
    def __init__(self,ratio = 0.5):
        self.reward_bias_ratio = ratio

global config_instance
config_instance = ConfigClass(ratio=0.1)
OUTPUT_DIR = os.path.join('data','output')
WORKLOAD_DATA_DIR = os.path.join('data','workload')
MODEL_PREDICTOR_DIR = os.path.join('data','model')
LONGTERM_WORKLOAD_DIR = os.path.join('data','longterm')
LONGTERM_EXACT_DIR = os.path.join('data','longterm_P80')
ONESTEP_WORKLOAD_DIR = os.path.join('data','onestep')
CACHE_DATA_DIR = os.path.join('data','cache')
BURSTY_CACHE_DIR = os.path.join(CACHE_DATA_DIR,'burst')
# OUTPUT_DIR = os.path.join('simulator','data','output')
# WORKLOAD_DATA_DIR = os.path.join('simulator','data','workload')
# MODEL_PREDICTOR_DIR = os.path.join('simulator','data','model')
# LONGTERM_WORKLOAD_DIR = os.path.join('simulator','data','longterm')
# ONESTEP_WORKLOAD_DIR = os.path.join('simulator','data','onestep')

def create_folder_if_not_exist(workload_path):
    Path(workload_path).mkdir(parents=True, exist_ok=True)

def parse_workload_data(workload_data, aim_mean = None, aim_std = None, des_mean = None, des_std = None):
    """
    aim_mean, aim_std 表示目标数组进行正则化时的参数
    des_mean, des_std 表示放缩后的目标参数
    """
    aim_workload =  (workload_data - aim_mean)  / aim_std * des_std + des_mean
    aim_workload[aim_workload<0] = 0
    # 最大值限制问题（可以考虑取消这个限制，然后把真实值加上）
    aim_workload[aim_workload>MAX_WORKLOAD_FACTOR*des_mean] = MAX_WORKLOAD_FACTOR*des_mean 
    return aim_workload

def read_workload_data(workload_name, des_mean = None, des_std = None):
    workload_file_name = workload_name + '.csv'
    workload_file = pd.read_csv(os.path.join(WORKLOAD_DATA_DIR,workload_file_name))
    workload_data = workload_file[['view']].values.ravel()  # workload_data np.array 存储流量信息
    if des_mean == None or des_std == None:
        return workload_data
    else:
        return parse_workload_data(workload_data, des_mean, des_std)

def read_longterm_prediction_result(workload_name,workload_dir = LONGTERM_WORKLOAD_DIR):
    """
        读取长期预测的结果
        写入的位置为长期预测器内的/model_framework/informer_model.py的InformerModel的adp_test方法，由adp_train命令指导开启入口
    """
    workload_file_dir = os.path.join(workload_dir, workload_name)
    origin_data = np.load(os.path.join(workload_file_dir, 'origin_data.npy'))   # (ts,1) 对应的原始流量数据，用来进行检验
    origin_timestamp = np.load(os.path.join(workload_file_dir, 'origin_timestamp.npy')) # (ts) 对应的是时间戳数据，用来产生tof
    # 备注： 以下三个数据为经过异常处理后的结果，因此会有一定的误差，仅作为分析使用。
    # 正确的流量判断需要基于原始变量数据进行
    origin = np.load(os.path.join(workload_file_dir, 'origin.npy')) #  (ts, seq_len) 输入预测器的流量信息数据
    pred = np.load(os.path.join(workload_file_dir, 'pred.npy')) # (ts,pred_len,1+quantile_num) 长期预测的结果 
    true = np.load(os.path.join(workload_file_dir, 'true.npy')) # (ts,pred_len,1) 长期预测的实际结果
    return origin_data, origin_timestamp, origin, pred, true

def read_onestep_prediction_result(workload_name):
    """
        读取单步预测的结果, 其中与基础信息相关的部分
        写入的位置为长期预测器内的/model_framework/informer_model.py的InformerModel的adp_test方法，由adp_train命令指导开启入口
    """
    workload_file_dir = os.path.join(ONESTEP_WORKLOAD_DIR, workload_name)
    origin_data = np.load(os.path.join(workload_file_dir, 'origin_data.npy'))   # (ts,1) 对应的原始流量数据，用来进行检验
    origin_timestamp = np.load(os.path.join(workload_file_dir, 'origin_timestamp.npy')) # (ts) 对应的是时间戳数据，用来产生tof
    # 备注： 以下三个数据为经过异常处理后的结果，因此会有一定的误差，仅作为分析使用。
    # 正确的流量判断需要基于原始变量数据进行
    origin = np.load(os.path.join(workload_file_dir, 'origin.npy')) #  (ts, seq_len) 输入预测器的流量信息数据
    pred = np.load(os.path.join(workload_file_dir, 'pred.npy')) # (ts,1) 长期预测的结果 
    true = np.load(os.path.join(workload_file_dir, 'true.npy')) # (ts,1) 长期预测的实际结果
    return origin_data, origin_timestamp, origin, pred, true


def get_cpu_predictor():
    """
        读取CPU预测器，因为是唯一的所以不需要标识
    """
    return load(os.path.join(MODEL_PREDICTOR_DIR,'cpu_predictor.joblib'))

def get_res_predictor():
    """
        读取响应时间的预测器，因为是唯一的所以不需要标识
    """
    return load(os.path.join(MODEL_PREDICTOR_DIR,'res_predictor.joblib'))

def read_profilng_data(file_name):
    # 组成文件路径
    file_path = os.path.join('data','profiling',file_name)
    data = pd.read_csv(file_path)
    return data

def get_folder_name(workload_name, env_name,use_burst=False):
    folder_name = env_name + '_' + workload_name
    if use_burst:
        folder_name += '_burst'
    return folder_name
# actor的保存
counter = 0
def save_actor_model(workload_name, env_name, actor,use_burst=False, use_continuous = False):
    global counter
    folder_name = get_folder_name(workload_name,env_name,use_burst=use_burst)+f"reward{config_instance.reward_bias_ratio}"
    if use_continuous:
        folder_name += "_continuous"
    create_folder_if_not_exist(os.path.join(MODEL_PREDICTOR_DIR, folder_name))
    act_save_path = os.path.join(MODEL_PREDICTOR_DIR, folder_name,'actor.pth')
    torch.save(actor.state_dict(), act_save_path)

    # folder_name += str(counter)
    # counter += 1
    # create_folder_if_not_exist(os.path.join(MODEL_PREDICTOR_DIR, folder_name))
    # act_save_path = os.path.join(MODEL_PREDICTOR_DIR, folder_name,'actor.pth')
    # torch.save(actor.state_dict(), act_save_path)

def load_actor_model(workload_name, env_name,use_burst=False, use_continuous = False,counter="", if_outside=False):
    folder_name = get_folder_name(workload_name,env_name,use_burst=use_burst)+f"reward{config_instance.reward_bias_ratio}"
    if use_continuous:
        folder_name += "_continuous"
    folder_name += counter
    act_save_path = os.path.join(MODEL_PREDICTOR_DIR, folder_name,'actor.pth')
    if if_outside:
        act_save_path = os.path.join('simulator',act_save_path)
    return torch.load(act_save_path)

# critic的读取和保存
def save_critic_model(workload_name, env_name, critic,use_burst=False, use_continuous = False):
    folder_name = get_folder_name(workload_name,env_name,use_burst=use_burst)+f"reward{config_instance.reward_bias_ratio}"
    if use_continuous:
        folder_name += "_continuous"
    create_folder_if_not_exist(os.path.join(MODEL_PREDICTOR_DIR, folder_name))
    cri_save_path = os.path.join(MODEL_PREDICTOR_DIR, folder_name,'critic.pth')
    torch.save(critic.state_dict(), cri_save_path)

def load_critic_model(workload_name, env_name,use_burst=False, use_continuous = False):
    folder_name = get_folder_name(workload_name,env_name,use_burst=use_burst)+f"reward{config_instance.reward_bias_ratio}"
    if use_continuous:
        folder_name += "_continuous"
    cri_save_path = os.path.join(MODEL_PREDICTOR_DIR, folder_name,'critic.pth')
    return torch.load(cri_save_path)
