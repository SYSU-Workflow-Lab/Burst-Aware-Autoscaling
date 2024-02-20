# M/M/c队列
# 参考Efficient resource provisioning for elastic Cloud services based on machine learning techniques

import math
import logging

from util.constdef import TRAIN_MODE, VALI_MODE, TEST_MODE
from baseline_method.prediction_base_util import prepare_prediction_data

import numpy as np

def get_average_response_time(ins_num, arrival_rate,service_rate=100):
    # M/M/n
    c = ins_num
    utilization_factor = arrival_rate / (ins_num * service_rate)
    cc = 1
    for i in range(c):
        cc *= c/(c-i)
    p0_cache = cc * math.pow(utilization_factor,c) / (1-utilization_factor)
    p0_cache2 = 0
    for n in range(1,c):
        cache = 1
        for i in range(n):
            cache *= c/(n-i)
        cache *= math.pow(utilization_factor,n)
        p0_cache2 += cache
    idle_probability = 1 / (p0_cache + p0_cache2)
    if idle_probability>=1:
        idle_probability=1
    queue_waiting_time = cc * math.pow(utilization_factor,c+1) * idle_probability / (arrival_rate*math.pow(1-utilization_factor,2))
    return queue_waiting_time + 1/service_rate # 单位为秒

def get_avg_res(arrival_rate,service_rate=100):
    # M/M/1
    utilization_factor = arrival_rate / service_rate
    # L_q = 1 / service_rate / (1-utilization_factor)
    L_q = utilization_factor / (1-utilization_factor) / service_rate
    return L_q + 1/service_rate

def allocate_based_on_queue(dataStore, workload_name,
    train_step=10000, vali_step=3000, test_step=3000, 
    sla=8., des_mean=10000, des_std=3000):
    """
        对比算法复现
        Burst-Aware Predictive Autoscaling for Containerized Microservices, 2020 TSC, M. Abdullah etc.
        具体来说，是使用EN regression对流量进行预测，使用DTR根据预测结果指定实例数，
        然后根据一个burst-aware的机制，通过方差来判断当前是否处于burst状态

        超参数选择：k=10
        Args:
            dataStore，记录用的pd.DataFrame
    """
    logging.info(f"m/m/1 queue auto-scaling method for {workload_name}")
    env = prepare_prediction_data(workload_name=workload_name, train_step=train_step, vali_step=vali_step, test_step=test_step,
        sla=sla, des_mean=des_mean, des_std=des_std)
    # 获取单步预测器的结果
    train_workload, test_workload = env.train_workload_data, env.test_workload_data
    train_prediction, test_prediction = env.train_onestep_prediction, env.test_onestep_prediction
    test_workload = test_workload[env.stack_point_num-1:]

    # TODO 计算出对应的单位流量
    arr = 0
    while arr < 100:
        res = get_avg_res(arr)
        if res > sla:
            arr -= 1
            break
        arr += 1
    # Part1: 训练最佳实例数，训练预测器
    # test_optimal_ins_array = np.ceil(test_prediction / optimal_value)
    test_ins_array = np.zeros_like(test_prediction)
    for i in range(len(test_ins_array)):
        test_ins_array[i] = math.ceil(test_prediction[i] / arr) # 根据预测结果来分配实例
    test_optimal_array = np.zeros_like(test_workload)
    for i in range(len(test_optimal_array)):
        test_optimal_array[i] = env.calculate_max_reward_instance(test_workload[i])
    # 使用训练部分的数据，训练得到一个决策树回归器，输入的是流量数，输出的是实例数。
         
    # 最终对结果进行验证和判断。
    # 验证段和测试段。如果有需要burst的话也是在这里
    res_array = np.zeros_like(test_ins_array)
    cpu_array = np.zeros_like(test_ins_array) 
    for i in range(len(res_array)):
        per_workload = test_workload[i] / test_ins_array[i]
        res_array[i] = env.res_predictor.predict_once(per_workload,use_sampling=True)
        cpu_array[i] = env.cpu_predictor.predict_once(per_workload,use_sampling=True)
    remark = "baseline/onestep-prediction"
    setting = ""
    # 记录验证部分数据
    # TODO 未检查
    # 记录验证部分数据
    test_actual_array = test_ins_array.ravel()
    test_optimal_ins_array = test_optimal_array
    res_diff = res_array-sla
    res_diff[res_diff<0] = 0
    # VALI_MODE
    burst_array = env.burst_detector.burst_array
    burst_array = burst_array[env.train_step:env.train_step+env.vali_step]
    dataStore = dataStore.append({
        "remark":remark,
        "setting":setting, # 与validate结果保存相关的setting
        "workload_name":workload_name,# 流量名称
        "env_name": "onestep-prediction",     # 使用的环境名称（方法）
        "cur_mode": VALI_MODE,     # 当前所属的阶段
        "total_reward":0, # 强化学习的奖励值。如果不涉及到强化学习相关逻辑则设置为0
        "sla": np.sum(res_array[:vali_step][burst_array==1]>sla),          # sla违约总数
        "sla_total":np.sum(res_diff[:vali_step][burst_array==1]),
        "avg_utilization":np.average(cpu_array[:vali_step][burst_array==1]),
        "instance":np.sum(test_actual_array[:vali_step][burst_array==1]),     # 使用的总实例数
        "optimal_instance":np.sum(test_optimal_ins_array[:vali_step][burst_array==1]), # 理想实例数
        "over_instance": np.sum((test_actual_array[:vali_step] - test_optimal_ins_array[:vali_step])[burst_array==1].clip(min=0)), # 超供量
        "under_instance": np.sum((test_optimal_ins_array[:vali_step] - test_actual_array[:vali_step])[burst_array==1].clip(min=0)), # 不足量
        "steps":np.sum(burst_array==1),        # 结果中的总步长数
        "onestep_predict": 0, # 使用单步预测的比例
        "longterm_predict": 0, # 使用多步预测的比例
        "addition_info2":"burst",
    }, ignore_index=True)
    dataStore = dataStore.append({
        "remark":remark,
        "setting":setting, # 与validate结果保存相关的setting
        "workload_name":workload_name,# 流量名称
        "env_name": "onestep-prediction",     # 使用的环境名称（方法）
        "cur_mode": VALI_MODE,     # 当前所属的阶段
        "total_reward":0, # 强化学习的奖励值。如果不涉及到强化学习相关逻辑则设置为0
        "sla": np.sum(res_array[:vali_step][burst_array==0]>sla),          # sla违约总数
        "sla_total":np.sum(res_diff[:vali_step][burst_array==0]),
        "avg_utilization":np.average(cpu_array[:vali_step][burst_array==0]),
        "instance":np.sum(test_actual_array[:vali_step][burst_array==0]),     # 使用的总实例数
        "optimal_instance":np.sum(test_optimal_ins_array[:vali_step][burst_array==0]), # 理想实例数
        "over_instance": np.sum((test_actual_array[:vali_step] - test_optimal_ins_array[:vali_step])[burst_array==0].clip(min=0)), # 超供量
        "under_instance": np.sum((test_optimal_ins_array[:vali_step] - test_actual_array[:vali_step])[burst_array==0].clip(min=0)), # 不足量
        "steps":np.sum(burst_array==0),        # 结果中的总步长数
        "onestep_predict": 0, # 使用单步预测的比例
        "longterm_predict": 0, # 使用多步预测的比例
        "addition_info2":"non-burst",
    }, ignore_index=True)
    dataStore = dataStore.append({
        "remark":remark,
        "setting":setting, # 与validate结果保存相关的setting
        "workload_name":workload_name,# 流量名称
        "env_name": "onestep-prediction",     # 使用的环境名称（方法）
        "cur_mode": VALI_MODE,     # 当前所属的阶段
        "total_reward":0, # 强化学习的奖励值。如果不涉及到强化学习相关逻辑则设置为0
        "sla": np.sum(res_array[:vali_step]>sla),          # sla违约总数
        "sla_total":np.sum(res_diff[:vali_step]),
        "avg_utilization":np.average(cpu_array[:vali_step]),
        "instance":np.sum(test_actual_array[:vali_step]),     # 使用的总实例数
        "optimal_instance":np.sum(test_optimal_ins_array[:vali_step]), # 理想实例数
        "over_instance": np.sum((test_actual_array[:vali_step] - test_optimal_ins_array[:vali_step]).clip(min=0)), # 超供量
        "under_instance": np.sum((test_optimal_ins_array[:vali_step] - test_actual_array[:vali_step]).clip(min=0)), # 不足量
        "steps":vali_step,        # 结果中的总步长数
        "onestep_predict": 0, # 使用单步预测的比例
        "longterm_predict": 0, # 使用多步预测的比例
        "addition_info2":"total",
    }, ignore_index=True)
    # TEST_MODE
    burst_array = env.burst_detector.burst_array
    burst_array = burst_array[env.train_step+env.vali_step:]
    dataStore = dataStore.append({
        "remark":remark,
        "setting":setting, # 与validate结果保存相关的setting
        "workload_name":workload_name,# 流量名称
        "env_name": "onestep-prediction",     # 使用的环境名称（方法）
        "cur_mode": TEST_MODE,     # 当前所属的阶段
        "total_reward":0, # 强化学习的奖励值。如果不涉及到强化学习相关逻辑则设置为0
        "sla": np.sum(res_array[vali_step:][burst_array==1]>sla),          # sla违约总数
        "sla_total":np.sum(res_diff[vali_step:][burst_array==1]),
        "avg_utilization":np.average(cpu_array[vali_step:][burst_array==1]),
        "instance":np.sum(test_actual_array[vali_step:][burst_array==1]),     # 使用的总实例数
        "optimal_instance":np.sum(test_optimal_ins_array[vali_step:][burst_array==1]), # 理想实例数
        "over_instance": np.sum((test_actual_array[vali_step:] - test_optimal_ins_array[vali_step:])[burst_array==1].clip(min=0)), # 超供量
        "under_instance": np.sum((test_optimal_ins_array[vali_step:] - test_actual_array[vali_step:])[burst_array==1].clip(min=0)), # 不足量
        "steps":np.sum(burst_array==1),        # 结果中的总步长数
        "onestep_predict": 0, # 使用单步预测的比例
        "longterm_predict": 0, # 使用多步预测的比例
        "addition_info2":"burst",
    }, ignore_index=True)
    dataStore = dataStore.append({
        "remark":remark,
        "setting":setting, # 与validate结果保存相关的setting
        "workload_name":workload_name,# 流量名称
        "env_name": "onestep-prediction",     # 使用的环境名称（方法）
        "cur_mode": TEST_MODE,     # 当前所属的阶段
        "total_reward":0, # 强化学习的奖励值。如果不涉及到强化学习相关逻辑则设置为0
        "sla": np.sum(res_array[vali_step:][burst_array==0]>sla),          # sla违约总数
        "sla_total":np.sum(res_diff[vali_step:][burst_array==0]),
        "avg_utilization":np.average(cpu_array[vali_step:][burst_array==0]),
        "instance":np.sum(test_actual_array[vali_step:][burst_array==0]),     # 使用的总实例数
        "optimal_instance":np.sum(test_optimal_ins_array[vali_step:][burst_array==0]), # 理想实例数
        "over_instance": np.sum((test_actual_array[vali_step:] - test_optimal_ins_array[vali_step:])[burst_array==0].clip(min=0)), # 超供量
        "under_instance": np.sum((test_optimal_ins_array[vali_step:] - test_actual_array[vali_step:])[burst_array==0].clip(min=0)), # 不足量
        "steps":np.sum(burst_array==0),        # 结果中的总步长数
        "onestep_predict": 0, # 使用单步预测的比例
        "longterm_predict": 0, # 使用多步预测的比例
        "addition_info2":"non-burst",
    }, ignore_index=True)
    dataStore = dataStore.append({
        "remark":remark,
        "setting":setting, # 与validate结果保存相关的setting
        "workload_name":workload_name,# 流量名称
        "env_name": "onestep-prediction",     # 使用的环境名称（方法）
        "cur_mode": TEST_MODE,     # 当前所属的阶段
        "total_reward":0, # 强化学习的奖励值。如果不涉及到强化学习相关逻辑则设置为0
        "sla": np.sum(res_array[vali_step:]>sla),          # sla违约总数
        "sla_total":np.sum(res_diff[vali_step:]),
        "avg_utilization":np.average(cpu_array[vali_step:]),
        "instance":np.sum(test_actual_array[vali_step:]),     # 使用的总实例数
        "optimal_instance":np.sum(test_optimal_ins_array[vali_step:]), # 理想实例数
        "over_instance": np.sum((test_actual_array[vali_step:] - test_optimal_ins_array[vali_step:]).clip(min=0)), # 超供量
        "under_instance": np.sum((test_optimal_ins_array[vali_step:] - test_actual_array[vali_step:]).clip(min=0)), # 不足量
        "steps":test_step,        # 结果中的总步长数
        "onestep_predict": 0, # 使用单步预测的比例
        "longterm_predict": 0, # 使用多步预测的比例
        "addition_info2":"total",
    }, ignore_index=True)
    return dataStore
