from baseline_method.prediction_base_util import prepare_prediction_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from util.constdef import TRAIN_MODE, TEST_MODE, VALI_MODE, OUTPUT_DIR, BURSTY_CACHE_DIR
import numpy as np
import logging
import os

def get_res_lower_bound_from_cpu(cpu_prdictor,res_predictor,cpu_lower_bound):
    """
        问题：目前已经有CPU的占用率下界，但是希望能获得响应时间对应的下界
        返回：响应时间下界
    """
    pass

def retrain_and_get_predict(train_optimal_ins_array, test_optimal_ins_array, allow_data_num):
    """
    Args:
        train_optimal_ins_array, 原有的初始训练数据
        test_optimal_ins_array, 后续的额外训练数据
    Returns:
        test_ins_array, 根据训练好的DTR得到的测试段数据的全部结果
    """
    pass

def threshold_validator(dataStore, workload_name, # 此部分超参数，参考/basic_env/predict_env.py中的PredictBasicEnv:__init__的设置
    train_step=10000, vali_step=3000, test_step=3000, 
    sla=8., des_mean=10000, des_std=3000):
    """
        阈值法的对比方法
    """
    logging.info(f"阈值法验证 for {workload_name}")
    cpu_lower_bound = 50 # 使用CPU占用率作为下阈值
    env = prepare_prediction_data(workload_name=workload_name, train_step=train_step, vali_step=vali_step, test_step=test_step,
        sla=sla, des_mean=des_mean, des_std=des_std)
    # 获取单步预测器的结果
    train_workload, test_workload = env.train_workload_data, env.test_workload_data
    stack_num = env.stack_point_num
    train_workload, test_workload = train_workload[stack_num-1:], test_workload[stack_num-1:]

    # Part1: 训练最佳实例数，训练预测器
    test_optimal_ins_array = np.zeros_like(test_workload)
    for i in range(len(test_workload)):
        test_optimal_ins_array[i] = env.calculate_max_reward_instance(test_workload[i])
    # 最终对结果进行验证和判断。
    # 验证段和测试段。如果有需要burst的话也是在这里
    test_actual_array = np.zeros_like(test_optimal_ins_array)
    res_array = np.zeros_like(test_actual_array)
    cpu_array = np.zeros_like(test_actual_array) 
    per_workload_list = list()
    cur_ins = env.calculate_max_reward_instance(train_workload[-1])
    
    for i in range(len(test_actual_array)):
        cur_workload = test_workload[i]
        per_workload = cur_workload / cur_ins
        res_array[i] = env.res_predictor.predict_once(per_workload,use_sampling=True)
        cpu_array[i] = env.cpu_predictor.predict_once(per_workload,use_sampling=True)
        test_actual_array[i] = cur_ins
        if res_array[i] >= sla:
            cur_ins += 1
        elif cpu_array[i] <= cpu_lower_bound:
            cur_ins -= 1
    remark = "baseline/threshold_based_scaler"
    setting = ""
    # 记录验证部分数据
    res_diff = res_array-sla
    res_diff[res_diff<0] = 0
    # NOTE 写入数据
    setting = f"threshold_autoscaling_result_{workload_name}_{VALI_MODE}"
    output_dir_name = os.path.join(OUTPUT_DIR, setting)
    if not os.path.exists(output_dir_name):
       os.mkdir(output_dir_name)
    np.save(os.path.join(output_dir_name, 'instance_array.npy'), test_actual_array[:vali_step])
    np.save(os.path.join(output_dir_name, 'workload_array.npy'), test_workload[:vali_step])
    np.save(os.path.join(output_dir_name, 'res_array.npy'), res_array[:vali_step])
    np.save(os.path.join(output_dir_name, 'cpu_array.npy'), cpu_array[:vali_step])
    setting = f"threshold_autoscaling_result_{workload_name}_{TEST_MODE}"
    output_dir_name = os.path.join(OUTPUT_DIR, setting)
    if not os.path.exists(output_dir_name):
       os.mkdir(output_dir_name)
    np.save(os.path.join(output_dir_name, 'instance_array.npy'), test_actual_array[vali_step:])
    np.save(os.path.join(output_dir_name, 'workload_array.npy'), test_workload[vali_step:])
    np.save(os.path.join(output_dir_name, 'res_array.npy'), res_array[:vali_step])
    np.save(os.path.join(output_dir_name, 'cpu_array.npy'), cpu_array[:vali_step])
    # VALI_MODE
    burst_array = env.burst_detector.burst_array
    burst_array = burst_array[env.train_step:env.train_step+env.vali_step]
    dataStore = dataStore.append({
        "remark":remark,
        "setting":setting, # 与validate结果保存相关的setting
        "workload_name":workload_name,# 流量名称
        "env_name": "burst-aware-prediction",     # 使用的环境名称（方法）
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
        "longterm_predict": np.average(np.abs(np.diff(test_actual_array[:vali_step][burst_array==1]))),
        "addition_info1":np.std(res_array[:vali_step][burst_array==1]),
        "addition_info2":"burst",
    }, ignore_index=True)
    dataStore = dataStore.append({
        "remark":remark,
        "setting":setting, # 与validate结果保存相关的setting
        "workload_name":workload_name,# 流量名称
        "env_name": "burst-aware-prediction",     # 使用的环境名称（方法）
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
        "longterm_predict": np.average(np.abs(np.diff(test_actual_array[:vali_step][burst_array==0]))),
        "addition_info1":np.std(res_array[:vali_step][burst_array==0]),
        "addition_info2":"non-burst",
    }, ignore_index=True)
    dataStore = dataStore.append({
        "remark":remark,
        "setting":setting, # 与validate结果保存相关的setting
        "workload_name":workload_name,# 流量名称
        "env_name": "burst-aware-prediction",     # 使用的环境名称（方法）
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
        "longterm_predict": np.average(np.abs(np.diff(test_actual_array[:vali_step]))),
        "addition_info1":np.std(res_array[:vali_step]),
        "addition_info2":"total",
    }, ignore_index=True)
    # TEST_MODE
    burst_array = env.burst_detector.burst_array
    burst_array = burst_array[env.train_step+env.vali_step:]
    dataStore = dataStore.append({
        "remark":remark,
        "setting":setting, # 与validate结果保存相关的setting
        "workload_name":workload_name,# 流量名称
        "env_name": "burst-aware-prediction",     # 使用的环境名称（方法）
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
        "longterm_predict": np.average(np.abs(np.diff(test_actual_array[vali_step:][burst_array==1]))),
        "addition_info1":np.std(res_array[vali_step:][burst_array==1]),
        "addition_info2":"burst",
    }, ignore_index=True)
    dataStore = dataStore.append({
        "remark":remark,
        "setting":setting, # 与validate结果保存相关的setting
        "workload_name":workload_name,# 流量名称
        "env_name": "burst-aware-prediction",     # 使用的环境名称（方法）
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
        "longterm_predict": np.average(np.abs(np.diff(test_actual_array[vali_step:][burst_array==0]))),
        "addition_info1":np.std(res_array[vali_step:][burst_array==0]),
        "addition_info2":"non-burst",
    }, ignore_index=True)
    dataStore = dataStore.append({
        "remark":remark,
        "setting":setting, # 与validate结果保存相关的setting
        "workload_name":workload_name,# 流量名称
        "env_name": "burst-aware-prediction",     # 使用的环境名称（方法）
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
        "longterm_predict": np.average(np.abs(np.diff(test_actual_array[vali_step:]))),
        "addition_info1":np.std(res_array[vali_step:]),
        "addition_info2":"total",
    }, ignore_index=True)
    return dataStore

def burst_aware_prediction_method(dataStore, workload_name, # 此部分超参数，参考/basic_env/predict_env.py中的PredictBasicEnv:__init__的设置
    train_step=10000, vali_step=3000, test_step=3000, 
    sla=8., des_mean=10000, des_std=3000):
    """
        对比算法复现
        Burst-Aware Predictive Autoscaling for Containerized Microservices, 2020 TSC, M. Abdullah etc.
        具体来说，是使用EN regression对流量进行预测，使用DTR根据预测结果指定实例数，
        然后根据一个burst-aware的机制，通过方差来判断当前是否处于burst状态

        超参数选择:k=10
        Args:
            dataStore，记录用的pd.DataFrame
    """
    logging.info(f"burst-aware prediction method for {workload_name}")
    k = 10
    env = prepare_prediction_data(workload_name=workload_name, train_step=train_step, vali_step=vali_step, test_step=test_step,
        sla=sla, des_mean=des_mean, des_std=des_std)
    # 获取单步预测器的结果
    train_workload, test_workload = env.train_workload_data, env.test_workload_data
    stack_num = env.stack_point_num
    train_workload, test_workload = train_workload[stack_num-1:], test_workload[stack_num-1:]
    train_prediction, test_prediction = env.train_onestep_prediction, env.test_onestep_prediction

    # Part1: 训练最佳实例数，训练预测器
    # TODO 使用响应式方法产生足够的数据。初步的想法是使用阈值方法\
    train_optimal_ins_array = np.zeros_like(train_workload)
    for i in range(len(train_optimal_ins_array)):
        train_optimal_ins_array[i] = env.calculate_max_reward_instance(train_workload[i])
    test_optimal_ins_array = np.zeros_like(test_workload)
    for i in range(len(test_workload)):
        test_optimal_ins_array[i] = env.calculate_max_reward_instance(test_workload[i])
    # train_optimal_ins_array = np.ceil(train_workload / optimal_value)
    # 使用训练部分的数据，训练得到一个决策树回归器，输入的是流量数，输出的是实例数。
    cpu_lower_bound = 50 # 响应式下降的下界
    train_reactive_ins_array = np.zeros_like(train_workload)
    cur_ins = env.calculate_max_reward_instance(train_workload[-1])
    cur_cpu = 0
    cur_res = 0
    st = False
    t = 0

    for i in range(len(train_workload)):
        cur_workload = train_workload[i]
        per_workload = cur_workload / cur_ins
        cur_res = env.res_predictor.predict_once(per_workload,use_sampling=True)
        cur_cpu = env.cpu_predictor.predict_once(per_workload,use_sampling=True)
        train_reactive_ins_array[i] = cur_ins
        if cur_res >= sla:
            cur_ins += 1
        elif cur_cpu <= cpu_lower_bound:
            cur_ins -= 1

    clf = DecisionTreeRegressor(random_state=0)
    clf.fit(train_workload.reshape(-1,1), train_reactive_ins_array.reshape(-1,1))
    test_ins_array_predict = np.ceil(clf.predict(test_prediction.reshape(-1,1)).ravel())

    # test_ins_array = np.ceil(test_prediction.ravel() / optimal_value)
    cache_test_prediction = test_prediction.ravel()
    test_ins_array = np.zeros_like(cache_test_prediction)
    for i in range(len(test_ins_array)):
        test_ins_array[i] = env.calculate_max_reward_instance(cache_test_prediction[i])
    # （然后在验证段和测试段进行一次检查，决策树回归暂定使用随机森林）
    assert len(test_ins_array) == vali_step + test_step

    test_ins_array = test_ins_array_predict
    # Part2：根据预测结果进行检查
    # 在测试段和验证段使用单步预测结果产生的实例数进行判断。
    if_using_new_burst = False # NOTE 是否使用最新版本的burst判断方式
    if if_using_new_burst:
        cache_directory_name = f"{workload_name}_{train_step}_{vali_step}_{test_step}"
        folder_path = os.path.join(BURSTY_CACHE_DIR,cache_directory_name)
        file_name = "is_burst.npy"
        is_burst = np.load(os.path.join(folder_path, file_name))
    else:
        is_burst = np.zeros_like(test_ins_array)

    isBurst = False # 当前是否处于burst状态
    past_ins = train_optimal_ins_array[-1] # 上一步的实例数
    current_ins = test_ins_array[0] # current_ins代表n_t+1，即当前即将应用上的实例（但还没有分配的实例）
    n_bb = 0
    backup_instance = train_optimal_ins_array[-k:] # 后备实例数，防止实例不够用的情况
    test_actual_array = np.zeros_like(test_ins_array)
    burst_array = np.zeros_like(test_ins_array)
    for i in range(len(test_ins_array)):
        current_ins = test_ins_array[i]
        if i-k<0:
            back_k_array = test_ins_array[:i+1]
        else:
            back_k_array = test_ins_array[i-k:i+1] # 必须包含i位置，总共应该有k+1个。第i个位置此时对应着未来的预测值
        if len(back_k_array) < k+1:
            back_k_array = np.concatenate([backup_instance,back_k_array])[-k-1:] # * 重点检查

        if not if_using_new_burst:
            # 使用原来的burst
            isBurst, current_ins, n_bb = judge_burst(isBurst, past_ins, current_ins, n_bb, back_k_array, k=k)
            if isBurst:
                is_burst[i] = 1
        else:
            isBurst = is_burst[train_step+i]
            if isBurst: # 如果当前处于burst状态，则将当前的实例数保存为之前的滑动窗口的最大值
                inst_max = current_ins
                for j in range(1,k+1):
                    if i-j>=0 and test_ins_array[i-j] > inst_max:
                        inst_max = test_ins_array[i-j]
                current_ins = inst_max

        burst_array[i] = isBurst
        test_actual_array[i] = current_ins
        past_ins = current_ins
         
    # 最终对结果进行验证和判断。
    # 验证段和测试段。如果有需要burst的话也是在这里
    res_array = np.zeros_like(test_ins_array)
    cpu_array = np.zeros_like(test_ins_array) 
    per_workload_list = list()
    for i in range(len(res_array)):
        if test_actual_array[i]==0:
            per_workload = env.MAX_WKL
        else:
            per_workload = test_workload[i] / test_actual_array[i]
        per_workload_list.append(per_workload)
        res_array[i] = env.res_predictor.predict_once(per_workload,use_sampling=True)
        cpu_array[i] = env.cpu_predictor.predict_once(per_workload,use_sampling=True)
    remark = "baseline/burst-aware-prediction"
    setting = ""
    # 记录验证部分数据
    res_diff = res_array-sla
    res_diff[res_diff<0] = 0
    # NOTE 写入数据
    setting = f"burst_aware_prediction_result_{workload_name}_{VALI_MODE}"
    output_dir_name = os.path.join(OUTPUT_DIR, setting)
    if not os.path.exists(output_dir_name):
       os.mkdir(output_dir_name)
    np.save(os.path.join(output_dir_name, 'instance_array.npy'), test_actual_array[:vali_step])
    np.save(os.path.join(output_dir_name, 'workload_array.npy'), test_workload[:vali_step])
    np.save(os.path.join(output_dir_name, 'res_array.npy'), res_array[:vali_step])
    np.save(os.path.join(output_dir_name, 'cpu_array.npy'), cpu_array[:vali_step])
    setting = f"burst_aware_prediction_result_{workload_name}_{TEST_MODE}"
    output_dir_name = os.path.join(OUTPUT_DIR, setting)
    if not os.path.exists(output_dir_name):
       os.mkdir(output_dir_name)
    np.save(os.path.join(output_dir_name, 'instance_array.npy'), test_actual_array[vali_step:])
    np.save(os.path.join(output_dir_name, 'workload_array.npy'), test_workload[vali_step:])
    np.save(os.path.join(output_dir_name, 'res_array.npy'), res_array[vali_step:])
    np.save(os.path.join(output_dir_name, 'cpu_array.npy'), cpu_array[vali_step:])
    # VALI_MODE
    burst_array = env.burst_detector.burst_array
    burst_array = burst_array[env.train_step:env.train_step+env.vali_step]
    burst_array = is_burst[:2000]

    dataStore = dataStore.append({
        "remark":remark,
        "setting":setting, # 与validate结果保存相关的setting
        "workload_name":workload_name,# 流量名称
        "env_name": "burst-aware-prediction",     # 使用的环境名称（方法）
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
        "longterm_predict": np.average(np.abs(np.diff(test_actual_array[:vali_step][burst_array==1]))),
        "addition_info1":np.std(res_array[:vali_step][burst_array==1]),
        "addition_info2":"burst",
    }, ignore_index=True)
    dataStore = dataStore.append({
        "remark":remark,
        "setting":setting, # 与validate结果保存相关的setting
        "workload_name":workload_name,# 流量名称
        "env_name": "burst-aware-prediction",     # 使用的环境名称（方法）
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
        "longterm_predict": np.average(np.abs(np.diff(test_actual_array[:vali_step][burst_array==0]))),
        "addition_info1":np.std(res_array[:vali_step][burst_array==0]),
        "addition_info2":"non-burst",
    }, ignore_index=True)
    dataStore = dataStore.append({
        "remark":remark,
        "setting":setting, # 与validate结果保存相关的setting
        "workload_name":workload_name,# 流量名称
        "env_name": "burst-aware-prediction",     # 使用的环境名称（方法）
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
        "longterm_predict": np.average(np.abs(np.diff(test_actual_array[:vali_step]))),
        "addition_info1":np.std(res_array[:vali_step]),
        "addition_info2":"total",
    }, ignore_index=True)
    # TEST_MODE
    burst_array = env.burst_detector.burst_array
    burst_array = burst_array[env.train_step+env.vali_step:]
    burst_array = is_burst[2000:]

    dataStore = dataStore.append({
        "remark":remark,
        "setting":setting, # 与validate结果保存相关的setting
        "workload_name":workload_name,# 流量名称
        "env_name": "burst-aware-prediction",     # 使用的环境名称（方法）
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
        "longterm_predict": np.average(np.abs(np.diff(test_actual_array[vali_step:][burst_array==1]))),
        "addition_info1":np.std(res_array[vali_step:][burst_array==1]),
        "addition_info2":"burst",
    }, ignore_index=True)
    dataStore = dataStore.append({
        "remark":remark,
        "setting":setting, # 与validate结果保存相关的setting
        "workload_name":workload_name,# 流量名称
        "env_name": "burst-aware-prediction",     # 使用的环境名称（方法）
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
        "longterm_predict": np.average(np.abs(np.diff(test_actual_array[vali_step:][burst_array==0]))),
        "addition_info1":np.std(res_array[vali_step:][burst_array==0]),
        "addition_info2":"non-burst",
    }, ignore_index=True)
    dataStore = dataStore.append({
        "remark":remark,
        "setting":setting, # 与validate结果保存相关的setting
        "workload_name":workload_name,# 流量名称
        "env_name": "burst-aware-prediction",     # 使用的环境名称（方法）
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
        "longterm_predict": np.average(np.abs(np.diff(test_actual_array[vali_step:]))),
        "addition_info1":np.std(res_array[vali_step:]),
        "addition_info2":"total",
    }, ignore_index=True)
    return dataStore

def judge_burst(isBurst, past_ins, current_ins, n_bb, back_k_array, k=10):
    """
        判断当前是否为Burst状态，并基于算法返回所需要的实例数
    在开始的时候，取得下一个时间点的流量和实例数
    对过去k个点的数据，计算最近的实例数上到过去第i个点的实例的方差（k=10）
            记录下最大的方差和最大的实例数
    1. 如果说，最大的方差比2大，且当前不是burst状态
        1. 当前实例数等于过去的最大值
        2. 设置为burst状态
        3. 记录下进入burst时的实例数
    2. 如果最大的方差比2大，且是burst状态
        1. 则保持最大值
    3. 如果最大的方差小于2，且是burst状态
        1. 当前当前的实例数小于进入burst时的实例数时
        2. burst状态取消
    4. 否则
        1. 正常分配
    """
    sigma_max = 0
    inst_max = 0
    for i in range(1,k+1):
        arr = back_k_array[-i-1:]
        sigma = np.std(arr)
        if sigma > sigma_max:
            sigma_max = sigma
            inst_max = np.max(arr)
    if sigma_max >= 2 and not isBurst:
        current_ins = inst_max
        n_bb = past_ins # 这个地方可能有点问题
        isBurst = True
    elif sigma >= 2 and isBurst:
        current_ins = inst_max
    elif sigma < 2 and isBurst:
        if current_ins < n_bb:
            # current_ins = n_t+1，值不变
            isBurst = False
            n_bb = 0
        else:
            current_ins = inst_max
    elif sigma < 2 and not isBurst:
        pass # current_ins = n_t+1，值不变
    else:
        raise NotImplementedError

    return isBurst, current_ins, n_bb