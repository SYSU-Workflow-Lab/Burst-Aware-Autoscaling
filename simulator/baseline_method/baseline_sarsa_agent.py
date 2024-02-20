# SARSA agent
# use SARSA-lambda to implement, source reference:https://github.com/naifmeh/smartbot/blob/master/sarsalambda.py
# doc: https://naifmehanna.com/2018-10-18-implementing-sarsa-in-python/

# 伪代码：
# 初始化SARSA(离散化环境)
# SARSA主过程（循环，直到终点）
# 1. 从sarsa_env获取当前的环境（离散的）
# 2. 根据环境结果，获取numpy数组的Q表
# 3. 使用episilon-greedy方法，选择动作
# 4. 应用选择的动作(step)，得到新的s
# 5. 更新Q表
# 6. 查看是否结束。如果未结束，则返回1；否则，进入7
# 7. terminate

# 不同方法的区别：SARSA状态的设计
# SARSA方法实现的细节
# 1. 超参数选择上。episilon的选择、Q值更新函数学习率的选择
from basic_env.reward_func import custom_reward
from util.constdef import ACTION_MAX, MODEL_PREDICTOR_DIR, TEST_MODE, TRAIN_MODE, VALI_MODE, get_folder_name, create_folder_if_not_exist
from basic_env.sarsa_env import ASARSA_Env

import os
import numpy as np
import random
import logging

from tqdm import trange
def get_action(Q_table,state,episilon,action_num=11):
    """
    采用基于episilon-greedy的方式产生动作
    """
    if random.random() < episilon:
        # epsilon
        action = random.randint(0,action_num-1)
    else:
        # greedy
        value = Q_table[tuple(state)]
        assert len(value.shape) == 1
        action = np.argmax(value)
    return action

def save_qtable(env,Q_table,workload_name,use_burst=False):
    env_name = env.env_name
    folder_name = get_folder_name(workload_name,env_name,use_burst=use_burst)
    create_folder_if_not_exist(os.path.join(MODEL_PREDICTOR_DIR, folder_name))
    act_save_path = os.path.join(MODEL_PREDICTOR_DIR, folder_name,'qtable.npy')
    np.save(act_save_path,Q_table)

def load_qtable(env,workload_name,use_burst=False):
    env_name = env.env_name
    folder_name = get_folder_name(workload_name,env_name,use_burst=use_burst)
    act_save_path = os.path.join(MODEL_PREDICTOR_DIR, folder_name,'qtable.npy')
    Q_table = np.load(act_save_path)
    return Q_table

def sarsa_asarsa_trainer(workload_name, action_max=10,
        train_step=6000, vali_step=4000, test_step=4000,
        use_burst=False, is_burst=False,sla=35.,des_mean=500, des_std=125):
    env = ASARSA_Env(workload_name=workload_name, action_max=action_max,
        train_step=train_step, vali_step=vali_step, test_step=test_step,
        use_burst=use_burst, is_burst=is_burst,sla=sla,des_mean=des_mean, des_std=des_std)
    Q_table = train_asarsa(env)
    save_qtable(env,Q_table,workload_name = workload_name,use_burst=use_burst)

def sarsa_asarsa_validator(dataStore, workload_name, action_max=10,
        train_step=6000, vali_step=4000, test_step=4000,
        use_burst=False, is_burst=False,sla=35.,des_mean=500, des_std=125):
    env = ASARSA_Env(workload_name=workload_name, action_max=action_max,
        train_step=train_step, vali_step=vali_step, test_step=test_step,
        use_burst=use_burst, is_burst=is_burst,sla=sla,des_mean=des_mean, des_std=des_std)
    Q_table = load_qtable(env, workload_name = workload_name, use_burst=use_burst)
    dataStore = validate_asarsa(dataStore, env,Q_table,cur_mode=VALI_MODE)
    dataStore = validate_asarsa(dataStore, env,Q_table,cur_mode=TEST_MODE)
    return dataStore

def train_asarsa(env):
    """
    Args:
    * 超参数设置集合，我觉得可以使用原predict_env代替
    """
    # 超参数设置部分
    discount = 0.9 # 计算TD-error，也可以写作gamma
    episilon = 0.1 # 动作选择episilon-greedy的超参数
    trace_decay = 0.9 # 用于资格迹的更新
    alpha = 0.01 # 学习率
    episode_num = int(3e5) # 3e5执行时间在6分钟左右

    batch_count = int(1e4)
    reward_list = list()
    reward_counter = 0
    
    early_stopping = True
    early_stop_limit = 10
    early_stop_count = 0

    cpu_number = env.cpu_state_num
    res_number = env.res_state_num
    max_action = ACTION_MAX
    action_number = 2 * max_action + 1
    Q_table = np.zeros((cpu_number,res_number,action_number))
    E_table = np.zeros((cpu_number,res_number,action_number)) # eligibility-trace

    Q_max_table = np.zeros_like(Q_table)
    reward_max = -np.inf
    # SARSA更新部分
    state = env.reset()
    action = get_action(Q_table,state,episilon,action_num=action_number)
    for e_i in trange(episode_num):
        next_state, reward, done, _ = env.step(action)
        next_action = get_action(Q_table,next_state,episilon,action_num=action_number)
        reward_counter += reward

        if done:
            q_target = reward
        else:
            q_target = reward + discount * Q_table[tuple(next_state)][next_action]
        q_predict =  Q_table[tuple(state)][action]
        td_error = q_target - q_predict
        E_table[tuple(state)][action] += 1

        Q_table += alpha * td_error * E_table
        E_table *= trace_decay * discount # accumulate method

        if done:
            state = env.reset()
            action = get_action(Q_table,state,episilon,action_num=action_number)
        else:
            state = next_state
            action = next_action
        
        # 一定时间段后打印统计的奖励值结果
        if e_i % batch_count == 0:
            if early_stopping:
                early_stop_count += 1
                if early_stop_count > early_stop_limit:
                    break
            reward_list.append(reward_counter)
            logging.info(f"{e_i} got avg reward {reward_counter / batch_count}")
            if reward_max < reward_counter:
                reward_max = reward_counter
                Q_max_table[:] = Q_table[:]
                early_stop_count = 0
            reward_counter = 0

    return Q_max_table
            
def validate_asarsa(dataStore, env, Q_table, cur_mode=VALI_MODE):
    """
    直接使用
    """
    state = env.reset(cur_mode=cur_mode)

    aim_steps = env.get_aim_step(cur_mode=cur_mode)
    instance_array = np.zeros(aim_steps)
    workload_array = np.zeros(aim_steps)
    cpu_array = np.zeros(aim_steps)
    res_array = np.zeros(aim_steps)
    action_array = np.zeros(aim_steps)
    reward_array = np.zeros(aim_steps)
    optimal_ins_array = np.zeros(aim_steps)
    optimal_workload_array = np.zeros(aim_steps)
    for i in range(len(optimal_ins_array)):
        cur_workload = env.get_latest_workload_by_point(i)
        optimal_workload_array[i] = cur_workload
        optimal_ins_array[i] = env.calculate_max_reward_instance(cur_workload)

    burst_array = env.burst_detector.burst_array
    if cur_mode == TRAIN_MODE:
        burst_array = burst_array[:env.train_step]
    elif cur_mode == VALI_MODE:
        burst_array = burst_array[env.train_step:env.train_step+env.vali_step]
    elif cur_mode == TEST_MODE:
        burst_array = burst_array[env.train_step+env.vali_step:]

    episilon = 0.0
    cpu_number = env.cpu_state_num
    res_number = env.res_state_num
    max_action = ACTION_MAX
    action_number = 2 * max_action + 1
    action = get_action(Q_table,state,episilon,action_num=action_number)
    # 定义数组
    for e_i in trange(aim_steps):
        next_state, reward, done, _ = env.step(action)
        next_action = get_action(Q_table,next_state,episilon,action_num=action_number)
        # 记录状态
        instance_array[e_i] = env.instance
        workload_array[e_i] = env.get_latest_workload_by_point()
        cpu_array[e_i] = env.get_cur_cpu()
        res_array[e_i] = env.get_cur_res()
        action_array[e_i] = action
        reward_array[e_i] = reward

        if done:
            state = env.reset()
            action = get_action(Q_table,state,episilon,action_num=action_number)
            break
        else:
            state = next_state
            action = next_action
    
    from util.constdef import OUTPUT_DIR
    workload_name = env.workload_name
    setting = f"arima_sarsa_result_{workload_name}_{cur_mode}"
    output_dir_name = os.path.join(OUTPUT_DIR, setting)
    if not os.path.exists(output_dir_name):
       os.mkdir(output_dir_name)
    np.save(os.path.join(output_dir_name, 'instance_array.npy'), instance_array)
    np.save(os.path.join(output_dir_name, 'workload_array.npy'), workload_array)
    np.save(os.path.join(output_dir_name, 'res_array.npy'), res_array)
    np.save(os.path.join(output_dir_name, 'cpu_array.npy'), cpu_array)
    # 记录在dataStore中
    remark = "baseline/asarsa_sarsa_version"
    setting = ""
    workload_name = env.workload_name
    res_sla = env.sla
    res_diff = res_array-res_sla
    res_diff[res_diff<0] = 0
    dataStore = dataStore.append({
        "remark":remark,
        "setting":setting, # 与validate结果保存相关的setting
        "workload_name":workload_name,# 流量名称
        "env_name":env.env_name,     # 使用的环境名称（方法）
        "cur_mode":cur_mode,     # 当前所属的阶段
        "total_reward":np.sum(reward_array), # 强化学习的奖励值。如果不涉及到强化学习相关逻辑则设置为0
        "sla": np.sum(res_array[burst_array==1]>res_sla),          # sla违约总数
        "sla_total":np.sum(res_diff[burst_array==1]),
        "avg_utilization":np.average(cpu_array[burst_array==1]),
        "instance":np.sum(instance_array[burst_array==1]),     # 使用的总实例数
        "optimal_instance":np.sum(optimal_ins_array[burst_array==1]), # 理想实例数
        "over_instance": np.sum((instance_array - optimal_ins_array)[burst_array==1].clip(min=0)), # 超供量
        "under_instance": np.sum((optimal_ins_array - instance_array)[burst_array==1].clip(min=0)), # 不足量
        "steps":np.sum(burst_array==1),        # 结果中的总步长数
        "addition_info2":"burst",
    }, ignore_index=True)
    dataStore = dataStore.append({
        "remark":remark,
        "setting":setting, # 与validate结果保存相关的setting
        "workload_name":workload_name,# 流量名称
        "env_name":env.env_name,     # 使用的环境名称（方法）
        "cur_mode":cur_mode,     # 当前所属的阶段
        "total_reward":np.sum(reward_array), # 强化学习的奖励值。如果不涉及到强化学习相关逻辑则设置为0
        "sla": np.sum(res_array[burst_array==0]>res_sla),          # sla违约总数
        "sla_total":np.sum(res_diff[burst_array==0]),
        "avg_utilization":np.average(cpu_array[burst_array==0]),
        "instance":np.sum(instance_array[burst_array==0]),     # 使用的总实例数
        "optimal_instance":np.sum(optimal_ins_array[burst_array==0]), # 理想实例数
        "over_instance": np.sum((instance_array - optimal_ins_array)[burst_array==0].clip(min=0)), # 超供量
        "under_instance": np.sum((optimal_ins_array - instance_array)[burst_array==0].clip(min=0)), # 不足量
        "steps":np.sum(burst_array==0),        # 结果中的总步长数
        "addition_info2":"non-burst",
    }, ignore_index=True)
    dataStore = dataStore.append({
        "remark":remark,
        "setting":setting, # 与validate结果保存相关的setting
        "workload_name":workload_name,# 流量名称
        "env_name":env.env_name,     # 使用的环境名称（方法）
        "cur_mode":cur_mode,     # 当前所属的阶段
        "total_reward":np.sum(reward_array), # 强化学习的奖励值。如果不涉及到强化学习相关逻辑则设置为0
        "sla": np.sum(res_array>res_sla),          # sla违约总数
        "sla_total":np.sum(res_diff),
        "avg_utilization":np.average(cpu_array),
        "instance":np.sum(instance_array),     # 使用的总实例数
        "optimal_instance":np.sum(optimal_ins_array), # 理想实例数
        "over_instance": np.sum((instance_array - optimal_ins_array).clip(min=0)), # 超供量
        "under_instance": np.sum((optimal_ins_array - instance_array).clip(min=0)), # 不足量
        "steps":aim_steps,        # 结果中的总步长数
        "addition_info2":"total",
    }, ignore_index=True)
    return dataStore
