# 为基于预测的对比算法提供基础性方法
from predictor.onestep_predictor import OneStepPredictor
from predictor.longterm_predictor import LongtermPredictor
from util.constdef import read_workload_data, parse_workload_data
from basic_env.predict_env import PredictBasicEnv
import numpy as np
import pandas as pd

def prepare_prediction_data(workload_name, # 此部分超参数，参考/basic_env/predict_env.py中的PredictBasicEnv:__init__的设置
    train_step=10000, vali_step=3000, test_step=3000, 
    sla=8., des_mean=10000, des_std=3000):
    """
        建议所有使用流量的对比算法使用此方法得到数据
        Args:
        * workload_name: 流量的名称，必须提供
        * start_point: 预测数据中的开始点
        * longterm_start_point: 长期预测数据中训练和验证所消耗的数据，需要跨库进行验证
        * stack_point_num: 主方法进行堆叠的数量，为了保证数据一致需要使用同样的参数
        * train_step/vali_step/test_step: 训练部分、验证部分和测试部分 所需要的时间点
        * des_mean/ des_std: 流量被放缩到的时间点。
        TODO 后续还需要限制流量的最大值，以降低模拟方法到现实方法迁移时的影响
    """
    env = PredictBasicEnv(workload_name=workload_name,sla=sla, 
        train_step=train_step, test_step=test_step, vali_step=vali_step,
        des_mean=des_mean, des_std=des_std,use_burst=True)
    return env
    # 对应的真实数据：env.train_workload_data, env.test_workload_data，其中前者长度为train_step，后者长度为vali_step+test_step
    # 对应的短期预测数据：env.train_onestep_prediction, env.test_onestep_prediction，(x,1)
    # 对应的长期预测数据：env.train_longterm_prediction, env.test_longterm_prediction，(x,24,3)
