"""
This project use source code from eRL_demo_PPOinSingleFile.py 
from [ElegantRL](https://github.com/AI4Finance-Foundation/ElegantRL), 
copyright Yonv1943曾伊言
licensed under the Apache 2.0 license. Followed by the whole Apache 2.0 license text.
"""
import os
import gym
import time
import math
import torch
from torch._C import TensorType
import torch.nn as nn
import numpy as np
import numpy.random as rd
from typing import Tuple
from copy import deepcopy
import pmdarima as pm
import pickle
from tqdm import trange
from util.constdef import MAX_INS, TOTAL_STEP, load_critic_model, load_actor_model, save_actor_model, save_critic_model, TRAIN_MODE, TEST_MODE, VALI_MODE, OUTPUT_DIR, create_folder_if_not_exist
from statsmodels.tsa.arima.model import ARIMA
import bootstrapped.bootstrap as bs
import functools
import logging

gym.logger.set_level(40)  # Block warning

class Arguments:
    def __init__(self, agent=None, env=None):
        self.agent = agent  # Deep Reinforcement Learning algorithm
        self.env = env  # the environment for training

        self.cwd = None  # current work directory. None means set automatically
        self.if_remove = False  # remove the cwd folder? (True, False, None:ask me)
        self.break_step = 2 ** 20  # break training after 'total_step > break_step'
        self.if_allow_break = True  # allow break training when reach goal (early termination)

        self.visible_gpu = '0'  # for example: os.environ['CUDA_VISIBLE_DEVICES'] = '0, 2,'
        self.worker_num = 2  # rollout workers number pre GPU (adjust it to get high GPU usage)
        self.num_threads = 8  # cpu_num for evaluate model, torch.set_num_threads(self.num_threads)

        '''Arguments for training'''
        self.gamma = 0.99  # discount factor of future rewards
        self.reward_scale = 2 ** 0  # an approximate target reward usually be closed to 256
        self.learning_rate = 2 ** -14  # 2 ** -14 ~= 6e-5
        self.soft_update_tau = 2 ** -8  # 2 ** -8 ~= 5e-3

        self.net_dim = 2 ** 9  # the network width
        self.batch_size = self.net_dim * 2  # num of transitions sampled from replay buffer.
        self.repeat_times = 2 ** 3  # collect target_step, then update network
        self.target_step = 2 ** 12  # repeatedly update network to keep critic's loss small
        # self.max_memo = self.target_step  # capacity of replay buffer
        self.max_memo = 2 ** 20
        self.if_per_or_gae = False  # GAE for on-policy sparse reward: Generalized Advantage Estimation.

        '''Arguments for evaluate'''
        self.eval_env = None  # the environment for evaluating. None means set automatically.
        self.eval_gap = 2 ** 6  # evaluate the agent per eval_gap seconds
        self.eval_times = 2  # number of times that get episode return in first
        self.random_seed = 0  # initialize random seed in self.init_before_training()

    def init_before_training(self, if_main):
        if self.cwd is None:
            agent_name = self.agent.__class__.__name__
            self.cwd = f'./{agent_name}_{self.env.env_name}_{self.visible_gpu}'

        if if_main:
            import shutil  # remove history according to bool(if_remove)
            if self.if_remove is None:
                self.if_remove = bool(input(f"| PRESS 'y' to REMOVE: {self.cwd}? ") == 'y')
            elif self.if_remove:
                shutil.rmtree(self.cwd, ignore_errors=True)
                print(f"| Remove cwd: {self.cwd}")
            # os.makedirs(self.cwd, exist_ok=True)

        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.set_num_threads(self.num_threads)
        torch.set_default_dtype(torch.float32)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.visible_gpu)

def ar_predict(coef, history):
    """
    manual ar predict
    https://machinelearningmastery.com/make-manual-predictions-arima-models-python/
    """
    yhat = 0.0
    for i in range(1,len(coef)+1):
        yhat += coef[i-1] * history[-i]
    return yhat

def loss_evaluate(ar_coef, train_data, alpha=0.99,num_iterations=50):
    """
    函数通过对训练部分代码进行误差分析，基于bootstrap方法得到一个置信区间的距离
    Args:
        ar_coef: AR2的参数
        train_data: 用来跑该参数的训练数据
        conf_ratio: 置信度
    Returns:
        val 表示真实值与置信区间上界的距离
    """
    loss_list = []
    for i in range(len(ar_coef),len(train_data)):
        pred = ar_predict(ar_coef, train_data[:i])
        loss_list.append(train_data[i]-pred)
    loss_list = np.array(loss_list)
    def upFunc(s, alpha=0.95):
        return np.percentile(np.asmatrix(s), (alpha + ((1.0 - alpha) / 2.0)) * 100, axis=1).ravel()

    # 使用偏函数固定除数据外的其他变量
    upBoundFunc = functools.partial(upFunc, alpha=alpha)
    high = bs.bootstrap(loss_list, stat_func=upBoundFunc, alpha=1 - alpha, num_iterations=num_iterations)
    return high.value

# TODO 完成burst部分的数据记录
def train_burst_scaler(dataStore, args,cur_mode=TRAIN_MODE,force_retrain = False):
    """
        args提供其他参数
        inner_env提供训练部分的数据
    """
    # 使用训练部分数据先跑出一个比较可靠的结果，或是直接预先指定
    # 声明一个ARMA模型
    inner_env = args.env.env
    previous_data = inner_env.norm_workload_data[:inner_env.longterm_cur_start_point]
    all_data = inner_env.norm_workload_data[inner_env.longterm_cur_start_point:inner_env.longterm_cur_start_point+inner_env.train_step+inner_env.vali_step+inner_env.test_step]
    # longterm_data[i][1]对应的是all_data[i-1]真值所对应的预测值
    onestep_data = inner_env.onestep_predictor.pred[inner_env.start_point+1:inner_env.start_point+inner_env.train_step+inner_env.vali_step+inner_env.test_step+1]
    longterm_data = inner_env.longterm_predictor.pred[inner_env.start_point+1:inner_env.start_point+inner_env.train_step+inner_env.vali_step+inner_env.test_step+1]
    # 由于burst部分是直接一起求的，所以使用bias来计算实际在每个段中的分布
    burst_array = inner_env.burst_detector.burst_array
    train_step = inner_env.train_step
    vali_step = inner_env.vali_step
    test_step = inner_env.test_step
    if cur_mode == TRAIN_MODE:
        start_point = 0
        first_point = inner_env.longterm_cur_start_point + start_point
        # all_data = inner_env.origin_workload_data[inner_env.longterm_cur_start_point:inner_env.longterm_cur_start_point+inner_env.train_step]
        aim_burst_array = inner_env.burst_detector.train_burst_array
        burst_array = burst_array[:train_step]
        index_bias = start_point
    elif cur_mode == VALI_MODE:
        start_point = inner_env.train_step
        first_point = inner_env.longterm_cur_start_point + start_point
        # all_data = inner_env.origin_workload_data[inner_env.longterm_cur_start_point+inner_env.train_step:inner_env.longterm_cur_start_point+inner_env.train_step+inner_env.vali_step]
        aim_burst_array = inner_env.burst_detector.vali_burst_array
        burst_array = burst_array[train_step:train_step+vali_step]
        index_bias = start_point
    elif cur_mode == TEST_MODE:
        start_point = inner_env.train_step + inner_env.vali_step
        first_point = inner_env.longterm_cur_start_point + start_point
        # all_data = inner_env.origin_workload_data[inner_env.longterm_cur_start_point+inner_env.train_step+inner_env.vali_step:inner_env.longterm_cur_start_point+inner_env.train_step+inner_env.vali_step+inner_env.test_step]
        aim_burst_array = inner_env.burst_detector.test_burst_array
        burst_array = burst_array[train_step+vali_step:]
        index_bias = start_point
    else:
        raise NotImplementedError
    # NOTE 记录数据
    total_aim_steps = inner_env.get_aim_step(cur_mode)
    instance_array = np.zeros(total_aim_steps)
    workload_array = np.zeros(total_aim_steps)
    cpu_array = np.zeros(total_aim_steps)
    res_array = np.zeros(total_aim_steps)
    predict_array = np.zeros(total_aim_steps)
    action_array = np.zeros(total_aim_steps)
    # 持久化保存相关参数
    remark = "burst-aware_burst"
    setting = remark + inner_env.generate_setting(without_ratio=True) + "_mode_" + str(cur_mode)
    path_dir = os.path.join(OUTPUT_DIR,setting)
    result_list_name = os.path.join(path_dir, "result_list.out")
    metric_list_name = os.path.join(path_dir, "metric_list.out")
    if not os.path.exists(path_dir) or force_retrain: # 如果不存在对应的目录或有强制训练标签，重新训练
        create_folder_if_not_exist(path_dir)
        min_length = 168 # 最少需要168的数据来训练，防止过拟合
        max_length = 336
        backup_data = inner_env.norm_workload_data[first_point-min_length:first_point]
        result_list = []
        metric_list = []
        print('start training ',cur_mode)
        for i in trange(len(aim_burst_array)):
            # 提取训练数据，使用前一段non-burst的数据训练模型。如果数据不足，则使用前一段burst的不足。如果超过则进行截断
            burst_start = aim_burst_array[i,0] + start_point
            burst_end = aim_burst_array[i,1] + start_point
            # prev_start和prev_end表示前一段burst区间的开始和结尾
            if i==0:
                prev_start = 0
                prev_end = 0 
            else:
                prev_start = aim_burst_array[i-1,0] + start_point
                prev_end = aim_burst_array[i-1,1]+1 + start_point
            train_data = all_data[prev_end:burst_start]
            if len(train_data) < min_length:
                if prev_start == burst_start: # 如果刚好是第一个，则使用准备好的备用数据
                    train_data = backup_data
                else:
                    train_data = all_data[:burst_start]
                    if len(train_data) < min_length and prev_start == 0:
                        rest_length = min_length - (burst_start - prev_start)
                        train_data = np.concatenate([previous_data[-rest_length:],train_data])
            if len(train_data) > max_length:
                train_data = train_data[-max_length:]
            # 使用训练部分跑一个结果
            train_data = np.nan_to_num(train_data)
            aim_avg = np.average(train_data)
            aim_std = np.std(train_data)
            aim_data = (train_data - aim_avg)/aim_std
            # model = pm.auto_arima(train_data, start_p=2, start_q=0, d=0, max_p=2, max_q=0,seasonal=False)
            default_coef = np.array([2,-1])
            try:
                model = ARIMA(aim_data,order=(2,0,0)).fit()
                ar_coef = model.arparams
            except Exception:
                ar_coef = default_coef

            onestep_result = []
            res_list = []
            ins_list = []
            reward_list = []
            max_instance = 0
            for ui in range(burst_start,burst_end+1):
                obs = ar_predict(ar_coef,aim_data)
                try:
                    model = ARIMA(aim_data[-max_length:],order=(2,0,0)).fit()
                    ar_coef = model.arparams
                except Exception:
                    ar_coef = default_coef
                
                up_threshold = loss_evaluate(ar_coef,aim_data[-max_length:])
                obs += up_threshold
                # 给定一个参数ar_coef，给定一个真实值的区间，给定一个置信度。目标：基于过去损失计算出一个置信区

                real_workload = all_data[ui] # 对应的真值
                up_thre = obs*aim_std + aim_avg # 估算的趋势预测值

                # 两道界限
                # 如果流量是下穿下阈值的，则
                diff_value = all_data[ui-5:ui-1] - longterm_data[ui-5:ui-1,0,0]
                if len(diff_value) > 0 and np.min(diff_value) < 0 and up_thre < longterm_data[ui,0,0]:
                    up_thre = longterm_data[ui,0,0]
                elif ( len(diff_value) == 0 or (len(diff_value) > 0 and np.min(diff_value) > 0)) and up_thre < longterm_data[ui,0,2]:
                    up_thre = longterm_data[ui,0,2]

                aim_data = np.append(aim_data,(all_data[ui]-aim_avg)/aim_std)
                onestep_result.append(up_thre)
                # 根据预测结果估算实例，然后记录响应时间、占用率等
                
                # TODO 需要在此处采用长期预测上阈值与当前预测值之间的最小数值
                instance = inner_env.calculate_max_reward_instance(up_thre)
                # instance = math.ceil(up_thre/aim_workload)
                # ! 改进：不下降
                if instance > MAX_INS:
                    instance = MAX_INS
                elif instance <= 0:
                    instance = 1

                current_cpu = inner_env.cpu_predictor.predict_once(real_workload/instance, instance = instance, use_sampling=True)
                current_res = inner_env.res_predictor.predict_once(real_workload/instance, instance = instance, use_sampling=True)
                reward = inner_env.get_reward(current_res,current_cpu)
                res_list.append(current_res)
                ins_list.append(instance)
                reward_list.append(reward)
                # 进行记录
                index = ui-index_bias
                instance_array[index] = instance
                cpu_array[index] = current_cpu
                res_array[index] = current_res
                workload_array[index] = real_workload
                predict_array[index] = up_thre

            result_list.append(np.array(onestep_result))
            metric_list.append([np.array(res_list),np.array(ins_list),np.array(reward_list)])
        # 持久化result_list
        with open(result_list_name,'wb') as fp:
            pickle.dump(result_list,fp)
        with open(metric_list_name,'wb') as fp:
            pickle.dump(metric_list,fp)

        np.save(os.path.join(path_dir, 'instance_array.npy'), instance_array)
        np.save(os.path.join(path_dir, 'workload_array.npy'), workload_array)
        np.save(os.path.join(path_dir, 'cpu_array.npy'), cpu_array)
        np.save(os.path.join(path_dir, 'res_array.npy'), res_array)
        np.save(os.path.join(path_dir, 'action_array.npy'), action_array)
        np.save(os.path.join(path_dir, 'predict_array.npy'), predict_array)
        np.save(os.path.join(path_dir, 'burst_array.npy'), burst_array)
    else:
        # 读取result_list和metric_list
        with open(result_list_name,'rb') as fp:
            result_list = pickle.load(fp)
        with open(metric_list_name,'rb') as fp:
            metric_list = pickle.load(fp)
        
    # 统计结果并输出
    res_count = 0
    rew_count = 0
    ins_count = 0
    total_step = 0
    
    for i in range(len(aim_burst_array)):
        res, ins, rew = metric_list[i]
        total_step += len(res)
        res_count += np.sum(res>inner_env.sla)
        ins_count += np.sum(ins)
        rew_count += np.sum(rew)

    dataStore = dataStore.append({
        "remark":remark,
        "setting":setting,                      # 与validate结果保存相关的setting
        "workload_name":args.workload_name,     # 流量名称
        "env_name":args.env.env.env_name,       # 使用的环境名称（方法）
        "cur_mode":cur_mode,                    # 当前所属的阶段
        "total_reward":rew_count,               # 强化学习的奖励值。如果不涉及到强化学习相关逻辑则设置为0
        "sla":np.sum(res_array>inner_env.sla),  # sla违约总数
        "instance":np.sum(instance_array),      # 使用的总实例数
        "steps":total_step,                     # 结果中的总步长数
        "onestep_predict":0,                    # 使用单步预测的比例
        "longterm_predict":0,                   # 使用多步预测的比例
    }, ignore_index=True)
    if total_step > 0:
        print(f"结果为：总奖励{rew_count}，平均奖励{rew_count/total_step}，SLA违约率{res_count/total_step} / 总SLA违约{res_count}，总实例{ins_count}，平均实例数{ins_count/total_step}，步长{total_step}")
    else:
        print(f"无burst-aware结果")
    return dataStore

def train_and_evaluate(args, agent_id=0,use_burst=False,use_two_step=False, load_data = False):
    """
    PPO时，if_off_policy=False
    DQN时，if_off_policy=True
    use_two_step 两步方法
    True时开启两步训练:
        第一步: 开启mask，开启的位置为agent开启，选择的时候不会产生对应的结果
            * agent.get_action
            * agent.forward
        第二步: 关闭mask，继续训练
    """
    # 核心函数，读取args参数并进行训练
    args.init_before_training(if_main=True)

    '''init: Agent'''
    env = args.env
    agent = args.agent
    agent.init(args.net_dim, env.state_dim, env.action_dim,
               args.learning_rate, args.if_per_or_gae) # 初始化Agent （PPOAgent）

    # DEBUG 读取保存数据
    # load_data = True
    # use_two_step = False
    # agent.act.if_use_multistep = True
    
    if load_data:
        agent.act.load_state_dict(load_actor_model(workload_name=args.env.workload_name,env_name=args.env.env_name,use_burst=use_burst))
        agent.cri.load_state_dict(load_critic_model(workload_name=args.env.workload_name,env_name=args.env.env_name,use_burst=use_burst))

    if use_two_step:
        args.break_step = int(args.env.max_step)+1 # 一个较小的量，可以设到5e4
        agent.act.is_action_mask = True
        agent.act.if_use_multistep = True

    '''init Evaluator'''
    eval_env = deepcopy(env)
    evaluator = Evaluator(args.cwd, agent_id, agent.device, eval_env, args.eval_times, args.eval_gap,use_burst=use_burst,use_continuous=not args.env.if_discrete)

    '''init ReplayBuffer'''
    buffer = list()

    def update_buffer(_trajectory):
        _trajectory = list(map(list, zip(*_trajectory)))  # 2D-list transpose
        ten_state = torch.as_tensor(_trajectory[0])
        ten_reward = torch.as_tensor(_trajectory[1], dtype=torch.float32) * reward_scale
        ten_mask = (1.0 - torch.as_tensor(_trajectory[2], dtype=torch.float32)) * gamma  # _trajectory[2] = done
        ten_action = torch.as_tensor(_trajectory[3])
        ten_noise = torch.as_tensor(_trajectory[4], dtype=torch.float32)

        buffer[:] = (ten_state, ten_action, ten_noise, ten_reward, ten_mask)

        _steps = ten_reward.shape[0]
        _r_exp = ten_reward.mean()
        return _steps, _r_exp

    '''start training'''
    # 正式开始训练
    cwd = args.cwd
    gamma = args.gamma
    break_step = args.break_step
    batch_size = args.batch_size # 默认是1024
    target_step = args.target_step
    reward_scale = args.reward_scale
    repeat_times = args.repeat_times
    if_allow_break = args.if_allow_break
    soft_update_tau = args.soft_update_tau

    agent.state = env.reset()

    if_train = True
    while if_train:
        # 通过探索环境获取足够的资格迹数据，并更新buffer
        with torch.no_grad():
            trajectory_list = agent.explore_env(env, target_step)
            steps, r_exp = update_buffer(trajectory_list)

        # 使用缓冲区数据更新网络
        logging_tuple = agent.update_net(buffer, batch_size, repeat_times, soft_update_tau)

        # 使用evaluator判断是否达到目标
        with torch.no_grad():
            # 训练阶段的数据仅作为参考使用
            if_reach_goal = evaluator.evaluate_and_save(agent.act,agent.cri,steps, r_exp, logging_tuple,cur_mode=TRAIN_MODE)
            # _ = evaluator.evaluate_and_save(agent.act,agent.cri,steps, r_exp, logging_tuple,cur_mode=TRAIN_MODE)
            if_train = not ((if_allow_break and if_reach_goal)
                            or evaluator.total_step > break_step
                            or os.path.exists(f'{cwd}/stop'))
            if not if_train and use_two_step and agent.act.is_action_mask:
                break_step = TOTAL_STEP
                agent.act.is_action_mask = False
                if_train = True
                past_max_r = evaluator.r_max
                evaluator = Evaluator(args.cwd, agent_id, agent.device, eval_env, args.eval_times, args.eval_gap,use_burst=use_burst,use_continuous=not args.env.if_discrete)
                evaluator.r_max = past_max_r
            # 允许停止的三个条件：（或条件）
            # 1. 允许结束且达到目标
            # 2. evaluator的评估步数达到停止步数(break_step)
            # 3. 路径中存在stop文件

    print(f'| UsedTime: {time.time() - evaluator.start_time:.0f} | SavedDir: {cwd}')


class Evaluator:
    def __init__(self, cwd, agent_id, device, env, eval_times, eval_gap,use_burst=False, use_continuous = False):
        self.recorder = list()  # total_step, r_avg, r_std, obj_c, ...
        self.recorder_path = f'{cwd}/recorder.npy'
        self.r_max = -np.inf
        self.total_step = 0
        self.use_burst = use_burst

        self.env = env
        self.cwd = cwd
        self.device = device
        self.agent_id = agent_id
        self.eval_gap = eval_gap
        # self.eval_times = eval_times
        self.eval_times = 1
        self.target_return = env.target_return
        self.workload_name = env.workload_name

        self.used_time = None
        self.use_continuous = use_continuous
        self.start_time = time.time()
        self.eval_time = 0
        self.earlystop_counter = 0
        self.earlystop_limit = 10
        self.allow_train_pass = True
        print(f"{'#' * 80}\n"
              f"{'ID':<3}{'Step':>10}{'maxR':>10} |"
              f"{'avgR':>8}{'stdR':>7}{'avgS':>7}{'stdS':>6} |"
              f"{'expR':>8}{'objC':>7}{'objA':>7}{'std_log':>7}|{'act_pre':>7}")

    def evaluate_and_save(self, act, cri, steps, r_exp, log_tuple, cur_mode=TRAIN_MODE) -> bool:
        """
            使用验证阶段的结果保存数据，使用训练阶段的结果显示训练成绩
        """
        self.total_step += steps  # update txuotal training steps

        # 如何利用好这个，使得验证时间不至于过短，又能不阻拦连续的两次探测
        if time.time() - self.eval_time < self.eval_gap: # 最大执行时间上界，默认为2**6
            return False 

        self.eval_time = time.time()
        rewards_steps_list = [get_episode_return_and_step(self.env, act, self.device,cur_mode,use_burst=self.use_burst) for _ in
                              range(self.eval_times)] # 此部分有self.eval_times控制，一般设置为2
        r_avg, r_std, s_avg, s_std = self.get_r_avg_std_s_avg_std(rewards_steps_list)

        if r_avg > self.r_max:  # 仅在vali_mode模式下执行保存
            self.earlystop_counter = 0
            self.r_max = r_avg  # update max reward (episode return)

            save_actor_model(self.workload_name, self.env.env_name, act,self.use_burst, self.use_continuous)
            save_critic_model(self.workload_name, self.env.env_name, cri,self.use_burst, self.use_continuous)
            print(f"{self.agent_id:<3}{self.total_step:10.2e}{self.r_max:10.2f} |")  # save policy and print
        else:
            self.earlystop_counter+=1

        self.recorder.append((self.total_step, r_avg, r_std, r_exp, *log_tuple))  # update recorder

        # 1. 上升期结束，比如超过10次调用validate没有发生一次最大值更新
        # 2. 达到预期目标（我不建议设置预期目标）
        if_reach_goal = bool(self.r_max > self.target_return)  # check if_reach_goal
        if self.earlystop_counter > self.earlystop_limit:
            logging.info("Early stopping")
            if_reach_goal = True
        if if_reach_goal and self.used_time is None:
            self.used_time = int(time.time() - self.start_time)
            print(f"{'ID':<3}{'Step':>10}{'TargetR':>10} |"
                  f"{'avgR':>8}{'stdR':>7}{'avgS':>7}{'stdS':>6} |"
                  f"{'UsedTime':>8}  ########\n"
                  f"{self.agent_id:<3}{self.total_step:8.2e}{self.target_return:8.2f} |"
                  f"{r_avg:8.2f}{r_std:7.1f}{s_avg:7.0f}{s_std:6.0f} |"
                  f"{self.used_time:>8}  ########")

        print(f"{self.agent_id:<3}{self.total_step:10.2e}{self.r_max:10.2f} |"
              f"{r_avg:8.2f}{r_std:7.1f}{s_avg:7.0f}{s_std:6.0f} |"
              f"{r_exp:8.2f}{''.join(f'{n:7.3f}' for n in log_tuple)}|{rewards_steps_list[0][-1]:7.3f}|t:{time.time()-self.start_time:.2f}")

        return if_reach_goal


    @staticmethod
    def get_r_avg_std_s_avg_std(rewards_steps_list):
        rewards_steps_ary = np.array(rewards_steps_list, dtype=np.float32)[...,:2]
        r_avg, s_avg = rewards_steps_ary.mean(axis=0)  # average of episode return and episode step
        r_std, s_std = rewards_steps_ary.std(axis=0)  # standard dev. of episode return and episode step
        return r_avg, r_std, s_avg, s_std
    

def get_episode_return_and_step(env, act, device,cur_mode=TRAIN_MODE,use_burst=False) -> Tuple[float, int]:
    """
        输入env, actor, device
        输出
        如果处于use_burst状态，系统会一直获取奖励，直到current_point变小
    """
    episode_return = 0.0  # sum of rewards in an episode
    episode_step = 1
    max_step = env.max_step
    if_discrete = env.if_discrete

    state = env.reset(cur_mode=cur_mode)
    use_predict_action = 0
    total_step = 0
    reward_list = []
    workload_list = []
    action_list = []
    ins_list = []
    res_list = []
    cpu_list = []

    for episode_step in range(max_step):
        s_tensor = torch.as_tensor((state,), device=device)
        a_tensor = act(s_tensor) # 耗时间1Ms
        if if_discrete:
            a_tensor = a_tensor.argmax(dim=1)
        action = a_tensor.detach().cpu().numpy()[0]  # not need detach(), because with torch.no_grad() outside
        if env.env.is_action_basedon_predict_result(action):
            use_predict_action += 1
        total_step += 1
        state, reward, done, _ = env.step(action) # 耗时间1Ms
        ins_list.append(env.instance) # 执行完动作后的实例数
        res_list.append(env.get_cur_res())
        cpu_list.append(env.get_cur_cpu())
        reward_list.append(reward)
        workload_list.append(env.get_latest_workload_by_point())
        action_list.append(action)
        episode_return += reward
        if done:
            if not use_burst: # 如果不是使用burst，则此时应该退出
                break
            else: # 如果使用burst，则应该继续进行枚举
                point1 = env.get_current_point()
                state = env.reset(cur_mode=cur_mode,restart=False)
                point2 = env.get_current_point()
                if point2 < point1: # 如果到达末尾，新的点会出现在最前面
                    break
    episode_return = getattr(env, 'episode_return', episode_return)
    return episode_return, episode_step, use_predict_action/total_step


class PreprocessEnv(gym.Wrapper):  # environment wrapper
    def __init__(self, env, if_print=True):
        # self.env = gym.make(env) if isinstance(env, str) else env
        if isinstance(env, str):
            raise NotImplementedError
        self.env = env
        self.workload_name = env.workload_name
        super().__init__(self.env)

        (self.env_name, self.state_dim, self.action_dim, self.action_max, self.max_step,
         self.if_discrete, self.target_return) = get_gym_env_info(self.env, if_print)
        self.env_name = env.env_name

    def reset(self, cur_mode=TRAIN_MODE,restart=True) -> np.ndarray:
        state = self.env.reset(cur_mode=cur_mode,restart=restart)
        return state.astype(np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        state, reward, done, info_dict = self.env.step(action * self.action_max) # action_max为保障action能取到最大值的ratio
        return state.astype(np.float32), reward, done, info_dict
    
    def get_current_point(self):
        return self.env.current_point

    def get_aim_step(self,cur_mode=TRAIN_MODE):
        return self.env.get_aim_step(cur_mode)


def get_gym_env_info(env, if_print) -> Tuple[str, int, int, int, int, bool, float]:
    assert isinstance(env, gym.Env)

    env_name = getattr(env, 'env_name', None)
    env_name = env.unwrapped.spec.id if env_name is None else None

    if isinstance(env.observation_space, gym.spaces.discrete.Discrete):
        raise RuntimeError("| <class 'gym.spaces.discrete.Discrete'> does not support environment with discrete observation (state) space.")
    state_shape = env.observation_space.shape
    state_dim = state_shape[0] if len(state_shape) == 1 else state_shape  # sometimes state_dim is a list

    target_return = getattr(env, 'target_return', None)
    target_return_default = getattr(env.spec, 'reward_threshold', None)
    if target_return is None:
        target_return = target_return_default
    if target_return is None:
        target_return = 2 ** 16

    max_step = getattr(env, 'max_step', None)
    max_step_default = getattr(env, '_max_episode_steps', None)
    if max_step is None:
        max_step = max_step_default
    if max_step is None:
        max_step = 2 ** 10

    if_discrete = isinstance(env.action_space, gym.spaces.Discrete)
    if if_discrete:  # make sure it is discrete action space
        action_dim = env.action_space.n
        action_max = int(1)
    elif isinstance(env.action_space, gym.spaces.Box):  # make sure it is continuous action space
        action_dim = env.action_space.shape[0]
        action_max = float(env.action_space.high[0])
        assert not any(env.action_space.high + env.action_space.low)
    else:
        raise RuntimeError('| Please set these value manually: if_discrete=bool, action_dim=int, action_max=1.0')

    print(f"\n| env_name:  {env_name}, action if_discrete: {if_discrete}"
          f"\n| state_dim: {state_dim}, action_dim: {action_dim}, action_max: {action_max}"
          f"\n| max_step:  {max_step:4}, target_return: {target_return}") if if_print else None
    return env_name, state_dim, action_dim, action_max, max_step, if_discrete, target_return

