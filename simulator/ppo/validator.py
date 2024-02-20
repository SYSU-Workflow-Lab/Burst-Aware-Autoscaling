# 验证已有的模型
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from ppo.common import Arguments, PreprocessEnv
from ppo.agent import AgentDiscretePPO
from util.constdef import LONGTERM_CONF_RATIO, TRAIN_MODE,TEST_MODE, VALI_MODE, load_actor_model, LONGTERM_START_POINT
from util.constdef import read_workload_data, config_instance
from util.constdef import OUTPUT_DIR, IF_COVER_BURST
from util.metric import SMAPE
from predictor.longterm_predictor import LongtermPredictor
from predictor.onestep_predictor import OneStepPredictor
from util.metric import QuantileLoss
from tqdm import trange
import pmdarima as pm

def generate_output_setting(args, workload_name, cur_mode, use_burst):
    """
        输入各种信息，
        输出一个字符串，代表着该场景的唯一情况
    """
    env = args.env.env
    env_name = env.env_name
    setting = f"{workload_name}_{cur_mode}_{use_burst}_burst_env_{env_name}_{env.generate_setting()}"
    return setting

def validate_actor(dataStore, args, workload_name, cur_mode=TEST_MODE, res_sla=8., 
    in_burst=False, use_burst=False, train_burst=0,remark="", if_on_policy=True):
    """
    Args:
    dataStore为记录所有数据的pandas对象
    函数的工作是再测试集合上验证已经得到的actor
    打印获得到的奖励列表、实例列表、状态列表和流量列表
    并且输出测试阶段的总奖励数
    train_burst表明数据的来源

    in_burst表示使用的模型是否为burst
    use_burst表示训练的模式是否为burst
    remark为写入到输出文件中的备注

    if_on_policy: PPO是on_policy的，DQN是off_policy的
    """
    torch.set_default_dtype(torch.float32)
    # init agent
    env = args.env
    args.agent.init(args.net_dim, env.state_dim, env.action_dim,
               args.learning_rate, args.if_per_or_gae) # 默认使用第一个GPU
    # load weight
    act = args.agent.act
    act.load_state_dict(load_actor_model(workload_name, env.env_name,use_burst=in_burst))
    act.eval()
    # try to get gpu
    gpu_id = 0
    device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
    # start validate
    episode_return = 0.0  # sum of rewards in an episode
    episode_step = 1
    max_step = env.max_step
    if_discrete = env.if_discrete
    state = env.reset(cur_mode=cur_mode,restart=True) 
    
    total_aim_steps = env.get_aim_step(cur_mode=cur_mode)
    # * 如果是non-burst-aware的话，则直接产生新数组
    optimal_ins_array = np.zeros(total_aim_steps)
    optimal_workload_array = np.zeros(total_aim_steps)
    for i in range(len(optimal_ins_array)):
        cur_workload = env.env.get_latest_workload_by_point(i)
        optimal_workload_array[i] = cur_workload
        optimal_ins_array[i] = env.env.calculate_max_reward_instance(cur_workload)
        
    if not use_burst:
        instance_array = np.zeros(total_aim_steps)
        workload_array = np.zeros(total_aim_steps)
        cpu_array = np.zeros(total_aim_steps)
        res_array = np.zeros(total_aim_steps)
        action_array = np.zeros(total_aim_steps)
    else:
        remark = "burst-aware_burst"
        setting = remark + env.env.generate_setting(without_ratio=True) + "_mode_" + str(cur_mode)
        path_dir = os.path.join(OUTPUT_DIR,setting)
        instance_array = np.load(os.path.join(path_dir, 'instance_array.npy'))
        workload_array = np.load(os.path.join(path_dir, 'workload_array.npy'))
        cpu_array = np.load(os.path.join(path_dir, 'cpu_array.npy'))
        res_array = np.load(os.path.join(path_dir, 'res_array.npy'))
        action_array = np.load(os.path.join(path_dir, 'action_array.npy'))
        assert np.sum(action_array) == 0
        action_array -= 1

    burst_array = env.env.burst_detector.burst_array
    if cur_mode == TRAIN_MODE:
        burst_array = burst_array[:env.env.train_step]
    elif cur_mode == VALI_MODE:
        burst_array = burst_array[env.env.train_step:env.env.train_step+env.env.vali_step]
    elif cur_mode == TEST_MODE:
        burst_array = burst_array[env.env.train_step+env.env.vali_step:]
    else:
        raise NotImplementedError
    instance_array[0] = env.env.instance
    workload_array[0] = env.env.get_latest_workload_by_point()
    cpu_array[0] = env.env.get_cur_cpu()
    res_array[0] = env.env.get_cur_res()
    action_array[0] = -1

    instance_list = list()
    reward_list = list()
    workload_list = list()
    cpu_list = list()
    res_list = list()
    violation_count = 0
    use_predictive_count = 0
    much_predictive_count = 0
    total_count = 0

    # burst侵入专用
    if_burst_over = False # 判断burst侵入是否结束（有->无信号）
    is_in_done = False # 判断burst侵入是否开始（无->有信号）
    for episode_step in range(max_step):
        s_tensor = torch.as_tensor((state,), device=device)
        a_tensor = act(s_tensor)
        if if_discrete:
            a_tensor = a_tensor.argmax(dim=1)
        action = a_tensor.detach().cpu().numpy()[0]  # not need detach(), because with torch.no_grad() outside
        if if_discrete:
            cur_action = action
        else:
            cur_action = np.where(np.max(action) == action)[0][0]
        if env.env.is_action_basedon_predict_result(action):
            use_predictive_count += 1
            if cur_action >= (2*env.env.action_max+1)+(2*env.env.pred_action_max+1):
                much_predictive_count += 1
        total_count += 1
        instance_list.append(env.env.instance) # 当前流量所对应的实例
        state, reward, done, info = env.step(action)
        currentResponseTime = env.env.get_cur_res()

        # if_burst_over = True # DEBUG
        # DEBUG 暂时去除，查看结果区别
        if currentResponseTime > res_sla:
            violation_count += 1
            if is_in_done:
                if_burst_over = True
        workload_list.append(env.env.get_latest_workload_by_point())
        reward_list.append(reward)
        cpu_list.append(env.env.get_cur_cpu())
        res_list.append(env.env.get_cur_res())

        # 记录
        current_step = info["step"]
        # * 这里需要选择是否覆盖掉burst的第一个点。burst的每一个点都是由之前的数据判断出来的，理论上不应该覆盖掉，但是实际上情况是比较复杂的
        if instance_array[current_step] == 0 or (is_in_done and not if_burst_over):
            instance_array[current_step] = env.env.instance
            workload_array[current_step] = env.env.get_latest_workload_by_point()
            cpu_array[current_step] = env.env.get_cur_cpu()
            res_array[current_step] = env.env.get_cur_res()
            action_array[current_step] = cur_action
            episode_return += reward
        if done or is_in_done:
            if not use_burst: # 如果不是使用burst，则此时应该退出
                break
            else: # 如果使用burst，则应该继续进行枚举
                if not IF_COVER_BURST: # 如果不做侵入，走这边
                    point1 = env.get_current_point()
                    state = env.reset(cur_mode=cur_mode,restart=False)
                    point2 = env.get_current_point()
                    if point2 < point1: # 如果到达末尾，新的点会出现在最前面
                        break

                    # NOTE 使用burst-aware/non-burst训练的时候，此处的env必须使用已有的实例
                    current_step = env.env.current_point
                    if instance_array[current_step] > 0: # 如果是burst下有的，直接进入
                        env.env.instance = instance_array[current_step] 
                    else: # 这里应该只有第一个会进入，应该是没有代码进来这个位置
                        pass
                    state = env.env._get_state_by_point(current_step, env.env.instance).astype(np.float32) # 这里承接其他位置
                # TODO 做burst侵入。单纯就执行而言，有两种情况
                else:
                    if not is_in_done:
                        is_in_done = True
                        if_burst_over = False
                    # 需要在此判断，是否到达下一个Burst的阶段
                    last_burst_point = env.env.burst_detector.get_next_non_burst_without_change(mode=cur_mode)-1 # NOTE 有待验证
                    if current_step >= total_aim_steps - 1 or current_step>=last_burst_point: # * NOTE 有待验证
                        if_burst_over = True

                    # 收到结束信号。
                    # 两种可能：1.无法继续下行；2. 前段发生了SLA，表明确实存在对应的风险。
                    if if_burst_over:
                        is_in_done = False
                        point1 = env.get_current_point()
                        state = env.reset(cur_mode=cur_mode,restart=False)
                        point2 = env.get_current_point()
                        if point2 < point1: # 如果到达末尾，新的点会出现在最前面
                            break

                        # NOTE 使用burst-aware/non-burst训练的时候，此处的env必须使用已有的实例
                        current_step = env.env.current_point
                        if instance_array[current_step] > 0: # 如果是burst下有的，直接进入
                            env.env.instance = instance_array[current_step] 
                        else: # 这里应该只有第一个会进入，应该是没有代码进来这个位置
                            pass
                        state = env.env._get_state_by_point(current_step, env.env.instance).astype(np.float32) # 这里承接其他位置

    episode_step += 1
    episode_return = getattr(env, 'episode_return', episode_return)
    res_diff = res_array-res_sla
    res_diff[res_diff<0] = 0
    # NOTE 1. 直接使用dataStore系统，向内部保存数据
    setting = generate_output_setting(args,workload_name,cur_mode,use_burst)
    dataStore = dataStore.append({
        "remark":remark,
        "setting":setting, # 与validate结果保存相关的setting
        "workload_name":workload_name,# 流量名称
        "env_name":args.env.env.env_name,     # 使用的环境名称（方法）
        "cur_mode":cur_mode,     # 当前所属的阶段
        "total_reward":episode_return, # 强化学习的奖励值。如果不涉及到强化学习相关逻辑则设置为0
        "sla": np.sum(res_array[burst_array==1]>res_sla),          # sla违约总数
        "sla_total":np.sum(res_diff[burst_array==1]),
        "avg_utilization":np.average(cpu_array[burst_array==1]),
        "instance":np.sum(instance_array[burst_array==1]),     # 使用的总实例数
        "optimal_instance":np.sum(optimal_ins_array[burst_array==1]), # 理想实例数
        "over_instance": np.sum((instance_array - optimal_ins_array)[burst_array==1].clip(min=0)), # 超供量
        "under_instance": np.sum((optimal_ins_array - instance_array)[burst_array==1].clip(min=0)), # 不足量
        "steps":np.sum(action_array==-1),        # 结果中的总步长数
        "onestep_predict":use_predictive_count/total_count, # 使用单步预测的比例
        # "longterm_predict":much_predictive_count/total_count, # 使用多步预测的比例
        "longterm_predict": np.average(np.abs(np.diff(instance_array[burst_array==1]))), 
        # "addition_info1":config_instance.reward_bias_ratio,
        "addition_info1":np.std(res_array[burst_array==1]),
        "addition_info2":"burst",
    }, ignore_index=True)
    dataStore = dataStore.append({
        "remark":remark,
        "setting":setting, # 与validate结果保存相关的setting
        "workload_name":workload_name,# 流量名称
        "env_name":args.env.env.env_name,     # 使用的环境名称（方法）
        "cur_mode":cur_mode,     # 当前所属的阶段
        "total_reward":episode_return, # 强化学习的奖励值。如果不涉及到强化学习相关逻辑则设置为0
        "sla": np.sum(res_array[burst_array==0]>res_sla),          # sla违约总数
        "sla_total":np.sum(res_diff[burst_array==0]),
        "avg_utilization":np.average(cpu_array[burst_array==0]),
        "instance":np.sum(instance_array[burst_array==0]),     # 使用的总实例数
        "optimal_instance":np.sum(optimal_ins_array[burst_array==0]), # 理想实例数
        "over_instance": np.sum((instance_array - optimal_ins_array)[burst_array==0].clip(min=0)), # 超供量
        "under_instance": np.sum((optimal_ins_array - instance_array)[burst_array==0].clip(min=0)), # 不足量
        "steps":np.sum(burst_array==0),        # 结果中的总步长数
        "onestep_predict":use_predictive_count/total_count, # 使用单步预测的比例
        "longterm_predict": np.average(np.abs(np.diff(instance_array[burst_array==0]))),
        "addition_info1":np.std(res_array[burst_array==0]),
        "addition_info2":"non-burst",
    }, ignore_index=True)
    dataStore = dataStore.append({
        "remark":remark,
        "setting":setting, # 与validate结果保存相关的setting
        "workload_name":workload_name,# 流量名称
        "env_name":args.env.env.env_name,     # 使用的环境名称（方法）
        "cur_mode":cur_mode,     # 当前所属的阶段
        "total_reward":episode_return, # 强化学习的奖励值。如果不涉及到强化学习相关逻辑则设置为0
        "sla": np.sum(res_array>res_sla),          # sla违约总数
        "sla_total":np.sum(res_diff),
        "avg_utilization":np.average(cpu_array),
        "instance":np.sum(instance_array),     # 使用的总实例数
        "optimal_instance":np.sum(optimal_ins_array), # 理想实例数
        "over_instance": np.sum((instance_array - optimal_ins_array).clip(min=0)), # 超供量
        "under_instance": np.sum((optimal_ins_array - instance_array).clip(min=0)), # 不足量
        "steps":total_aim_steps,        # 结果中的总步长数
        "onestep_predict":len(action_array[action_array>=2*env.env.action_max+1])/len(action_array), # 使用单步预测的比例
        "longterm_predict": np.average(np.abs(np.diff(instance_array))),
        "addition_info1":np.std(res_array),
        "addition_info2":"total",
    }, ignore_index=True)
    print(f"事实上采用burst动作的总数{np.sum(action_array==-1)}")
    # TODO 2. 需要保存对应的numpy数组，我的一个想法是根据设置生成一个唯一性的id（setting），然后用这个唯一性的id来进行检索
    # 该长度为对应的steps，即使没有（比如burst情况），长度也是固定的。
    output_dir_name = os.path.join(OUTPUT_DIR, setting)
    if not os.path.exists(output_dir_name):
       os.mkdir(output_dir_name)
    np.save(os.path.join(output_dir_name, 'instance_array.npy'), instance_array)
    np.save(os.path.join(output_dir_name, 'workload_array.npy'), workload_array)
    np.save(os.path.join(output_dir_name, 'cpu_array.npy'), cpu_array)
    np.save(os.path.join(output_dir_name, 'res_array.npy'), res_array)
    np.save(os.path.join(output_dir_name, 'action_array.npy'), action_array)
    if use_burst:
        np.save(os.path.join(output_dir_name, 'burst_array.npy'), burst_array)
    # 需要保存的数据：data_stamp
    # 唯一性id涉及到的内容：workload_name, 阶段, 采用的环境名， 是否采用burst模式， burst中的哪一个。
    # 如果是burst的话，这里只需要保存non-burst的结果，burst的时候自己将两者组合起来。主要是因为burst更容易发生改变。
    print(f"{episode_step}: got reward {episode_return}, prediction action {use_predictive_count/total_count:.4f}/{much_predictive_count/total_count:.4f}, sla violation count {violation_count/total_count:.4f}, total sla {violation_count} , with server number {np.mean(instance_list):.4f} and total server{np.sum(instance_list)}")
    print(f"总奖励{np.sum(reward_list)},总违约{violation_count}，总实例{np.sum(instance_list)}，总步长{total_count}")
    return dataStore, episode_return, episode_step

# 验证长期预测结果与短期预测的结果，基于SMAPE进行识别与处理，并将结果与实际流量进行结合
def workload_validator(workload_name, start_point=0, train_step=19000, test_step=100,seq_len=24):
    """
        根据给定的流量值，验证预测器的预测结果，指标为SMAPE
        start_point/train_step/test_step为模拟环境处给定的参数。
        seq_len为长期预测器的预测范围
    """
    origin_workload_data = read_workload_data(workload_name)
    longterm_predictor = LongtermPredictor(workload_name)
    onestep_predictor = OneStepPredictor(workload_name)
    # origin_data, origin_timestamp, origin, pred, true = read_longterm_prediction_result(workload_name)
    origin_timestamp = longterm_predictor.origin_timestamp
    longterm_start_point = LONGTERM_START_POINT # 由长期训练部分的len1+len2决定，如果有修改的话需要跨库进行对应的更正（
    start_point = start_point  # start_point 标识开始点
    current_point = 0  # current_point 标识当前点
    train_step = train_step
    test_step = test_step
    max_step = train_step
    longterm_cur_start_point = longterm_start_point + start_point
    assert longterm_cur_start_point + train_step + test_step <= len(origin_workload_data), "长度不会超标"
    assert start_point + train_step + test_step <= len(origin_timestamp), "长度不会超标"
    # 环境信息：此处必须往前一步，不然与预测值会对不上
    train_workload_data = origin_workload_data[longterm_cur_start_point-1:longterm_cur_start_point-1+train_step] 
    test_workload_data = origin_workload_data[longterm_cur_start_point-1+train_step:longterm_cur_start_point-1+train_step+test_step]
    # 时间戳信息
    train_timestamp = origin_timestamp[start_point:start_point+train_step]
    test_timestamp = origin_timestamp[start_point+train_step:start_point+train_step+test_step]
    # 预测信息
    train_longterm_prediction = longterm_predictor.pred[start_point:start_point+train_step]
    test_longterm_prediction = longterm_predictor.pred[start_point+train_step:start_point+train_step+test_step]
    train_onestep_prediction = onestep_predictor.pred[start_point:start_point+train_step]
    test_onestep_prediction = onestep_predictor.pred[start_point+train_step:start_point+train_step+test_step]
    onestep_result_validator(train_onestep_prediction, origin_workload_data[longterm_cur_start_point:longterm_cur_start_point+train_step])
    # longterm_result_validator(train_longterm_prediction[...,1], origin_workload_data[longterm_cur_start_point:longterm_cur_start_point+train_step+seq_len], seq_len=seq_len)
    # 对比实验1：ARIMA与多步预测实验
    transformer_result = longterm_threshold_result_validator(train_longterm_prediction, origin_workload_data[longterm_cur_start_point:longterm_cur_start_point+train_step+seq_len], seq_len=seq_len)
    print(np.mean(transformer_result))
    # arima_result = arima_threshold_result_validator(train_longterm_prediction, origin_workload_data[longterm_cur_start_point-168:longterm_cur_start_point+train_step+seq_len], origin_workload_data[longterm_cur_start_point:longterm_cur_start_point+train_step+seq_len])
    # print(np.mean(arima_result))
    # np.save('test.npy',arima_result)
    arima_result = np.load('test.npy')
    # 验证性实验
    # 下面考虑对置信区间进行分析，判断对burst的识别情况
    # 1. 绘制SMAPE图景
    # 2. 计算离群点距离上下阈值的最小值（在其中则为0）
    # 3. 研究预测质量（q-Risk）对结果的影响
    longterm_result_threshold_validator(train_longterm_prediction, origin_workload_data[longterm_cur_start_point:longterm_cur_start_point+train_step+seq_len], seq_len=seq_len)

def onestep_result_validator(onestep_result, real_result):
    result_list = []
    for i in range(len(onestep_result)):
        result_list.append(SMAPE(onestep_result[i].item(),real_result[i].item()))
    smape_val = SMAPE(onestep_result.ravel(),real_result.ravel())
    print(f"onestep prediction result is {smape_val}")
    plt.cla()
    plt.plot(result_list)
    plt.savefig('onestep_pred.png')
    plt.cla()
    plt.plot(real_result)
    plt.savefig('onestep_real.png')

def get_onestep_validate_result(onestep_result, real_result):
    result_list = []
    for i in range(len(onestep_result)):
        result_list.append(SMAPE(onestep_result[i].item(),real_result[i].item()))
    return np.array(result_list)

def get_longterm_validate_result(longterm_result, real_result):
    """
        longterm_result
        real_result 是单步的结果
        计算单步的长期预测值的q-risk结果，因此实际上与onestep的占用是一致的
    """
    quantile = LONGTERM_CONF_RATIO
    qrisk = QuantileLoss(quantiles=[1-quantile,0.5,quantile])
    result_list = []
    for i in range(len(longterm_result)):
        aim_data = longterm_result[i,:1]
        result_list.append(qrisk.numpy_normalised_quantile_loss(aim_data,real_result[i]))
    return np.array(result_list)

def longterm_result_validator(longterm_result, real_result, seq_len=24):
    result_list = []
    for i in range(len(longterm_result)):
        real_data = real_result[i:i+seq_len]
        result_list.append(SMAPE(longterm_result[i],real_data))
    smape_val = np.mean(result_list)
    print(f"longterm prediction result is {smape_val}")
    plt.cla()
    plt.plot(result_list)
    plt.savefig('longterm_pred.png')
    plt.cla()
    plt.plot(real_result)
    plt.savefig('longterm_real.png')

def longterm_threshold_result_validator(longterm_threshold_result, real_result, seq_len=24):
    quantile=LONGTERM_CONF_RATIO
    result_list = []
    qrisk = QuantileLoss(quantiles=[1-quantile,0.5,quantile])
    print("longterm_threshold_result_validator")
    for i in trange(len(longterm_threshold_result)):
        real_data = real_result[i:i+seq_len]
        result_list.append(qrisk.numpy_normalised_quantile_loss(longterm_threshold_result[i],real_data,quantile=quantile))
    smape_val = np.mean(result_list)
    print(f"longterm prediction result is {smape_val}")
    # plt.cla()
    # plt.plot(result_list)
    # plt.savefig('longterm_pred.png')
    # plt.cla()
    # plt.plot(real_result)
    # plt.savefig('longterm_real.png')
    return result_list

def arima_threshold_result_validator(aim_data, train_data, real_result,quantile=0.9):
    seq_len = 168 # 可以改
    train_len = 168 # 不可更改
    print("arima_threshold_result_validator")
    pred_count = (len(train_data)-168)//seq_len
    pred_result = np.zeros((aim_data.shape[0]+aim_data.shape[1]-1,3))
    for i in trange(pred_count):
        pivot = i * seq_len
        train_data_inneed = train_data[pivot:pivot+train_len]
        real_data = real_result[pivot:pivot+seq_len]
        preds = np.random.rand(seq_len,3)
        # preds = np.zeros_like(pred_result[pivot:pivot+seq_len])
        preds = train_arma_model_and_predict(train_data_inneed, real_data)
        pred_result[pivot:pivot+seq_len] = preds
    # 最后一次
    pivot = seq_len * pred_count
    train_data_inneed = train_data[pivot:pivot+train_len]
    real_data = real_result[pivot:-1]
    preds = train_arma_model_and_predict(train_data_inneed, real_data)
    pred_result[pivot:pivot+seq_len] = preds
    # 每24次进行一轮
    aim_len = 24
    result_list = []
    for i in range(len(aim_data)):
        result_list.append(pred_result[i:i+aim_len])
    longterm_threshold_result = np.array(result_list)

    result_list = []
    qrisk = QuantileLoss(quantiles=[1-quantile,0.5,quantile])
    print("longterm_threshold_result_validator")
    for i in trange(len(longterm_threshold_result)):
        real_data = real_result[i:i+aim_len]
        result_list.append(qrisk.numpy_normalised_quantile_loss(longterm_threshold_result[i],real_data,quantile=0.9))
    return result_list

def train_arma_model_and_predict(train_data, test_data, max_length=168*3):
    """
        ARMA的预测器，直接使用训练段跑数据看效果
        正则化：可以先取对数再做正则化，或者不进行正则化（可能有问题）
        Args:
            train_data为在此之前的non-burst数据
            test_data为当前的burst数据，滚动更新
    """
    # 归一化处理。取数据的P90 - P10做归一化
    aim_data = np.sort(train_data)
    aim_max, aim_min = aim_data[int(0.9 * len(aim_data))], aim_data[int(0.1 * len(aim_data))]
    train_data = (train_data - aim_min) / (aim_max - aim_min)
    test_data = (test_data - aim_min) / (aim_max - aim_min)
    if len(train_data) > max_length:
        train_data = train_data[-max_length:]
    model = pm.auto_arima(train_data, start_p=1, start_q=1, d=0, max_p=4, max_q=4)
    # 根据train部分预测的质量来进行分析
    # insample_predict_data, conf_int = model.predict_in_sample(start=0,end=len(train_data)-1,
    #                                                 return_conf_int=True)
    loss = QuantileLoss(quantiles=[0.1,0.5,0.9])
    # qrisk = loss.numpy_normalised_quantile_loss(np.stack([conf_int[:,0], insample_predict_data, conf_int[:,1]]).transpose(), train_data, quantile=0.9)
    result = np.zeros((len(test_data),3))
    for i in range(len(test_data)):
        # preds = model.predict(n_periods=1)
        preds, conf_int = model.predict(n_periods=1, alpha=0.1, # 返回95%置信区间的结果
                                return_conf_int=True)
        result[i,0] = conf_int[0,0].item()
        result[i,1] = preds[0].item()
        result[i,2] = conf_int[0,1].item()
        model.update(test_data[i])
    return result * (aim_max - aim_min) + aim_min

def calculate_outlier(preds, trues):
    """
        preds:(24,3)
        trues:(24)
        计算outlier的数量，以及偏离距离
        尝试用了下中文变量，说实话看着眼睛有点累。如果需要比较复杂的算法确实是更自由一些。（算了，还是换了回去，测起来太麻烦了）
    """
    up_thre, down_thre = preds[...,2], preds[...,0]
    outlier_num = np.sum(np.logical_or(up_thre < trues, down_thre > trues))
    up_outlier_dist = trues - up_thre
    down_outlier_dist = down_thre - trues
    up_outlier_dist[up_outlier_dist<0] = 0
    down_outlier_dist[down_outlier_dist<0] = 0
    outlier_dist_array = (up_outlier_dist + down_outlier_dist) / preds[...,1]
    assert np.sum(outlier_dist_array == 0) == len(trues) - outlier_num
    if len(outlier_dist_array[outlier_dist_array>0]) == 0:
        return outlier_num, np.sum(outlier_dist_array)
    else:
        return outlier_num, np.sum(outlier_dist_array)

def burst_detection(preds, trues, status = False, score_threshold = 0.3, num_score=0.1, avail_horizon=4):
    """
        给定所有包含这一个点的预测器的预测结果（应该为24个），以及这一段的真实值打包的结果
        status表示上一个点是否为burst状态
            对于一开始的24个点，默认为non-burst状态。对于之后的部分，如果没有找到，则使用true值进行替代
        Returns:
            返回当前点是否为burst状态
        注意：
            划定可用数据界。函数只能判断当前状态是否为burst状态，以i=23为例，所有超出23的真实值都应该予以忽略
        * 具体来说，第一个预测结果有24个点可用，第二个点有23个点可用，...最后一个预测结果只有一个点是可用的。不能将未来的情况纳入到现在考虑
        ? 有一个问题，如果长期预测器的表现是处于有问题的状态，这时候应该如何去处理？
        ? 具体来说，比如长期预测在4998左右会出现SMAPE的突变，突变的原因是异常值远远大于实际情况，击穿了localMinmax，干扰了正常的预测长达168个点。
        解决方案：引入avail_horizon有效区间。即我们只考虑一段那时间内的违约情况，如果超出了这个区间则不进行考虑。
            区间长度由经验决定，一般来说，从non-burst到burst是不需要这个的，而更多与从burst状态恢复相关。6个点一般来说是足够的，4个点其实也可以。
    """
    ts_length, pred_len, quantile_count = preds.shape
    assert ts_length == pred_len # 这两个值必须要相等
    assert ts_length == trues.shape[0] 
    assert pred_len == trues.shape[1]
    score_list = []
    num_list = []
    # 根据给定的这部分数据，计算每一个点的outlier值和距离，并给每个预测结果计算分数
    # 根据给定的分数进行判断
    # 如果上一个点处于non-burst状态，且有2个以上点违约，总分数大于0.3，则当前处于burst；反之则为non-burst
    # 如果上一个点处于burst状态，且所有的点都小于0.3，则为non-burst；反之则继续维持在burst上。
    for i in range(len(preds)):
        if len(preds)-i>avail_horizon:
            continue
        pred = preds[i,:pred_len-i,:]
        true = trues[i,:pred_len-i]
        up_thre, down_thre = pred[...,2], pred[...,0]
        outlier_num = np.sum(np.logical_or(up_thre < true, down_thre > true))
        num_list.append(outlier_num)

        up_outlier_dist = true - up_thre
        down_outlier_dist = down_thre - true
        up_outlier_dist[up_outlier_dist<0] = 0
        down_outlier_dist[down_outlier_dist<0] = 0
        outlier_dist_array = (up_outlier_dist + down_outlier_dist) / pred[...,1]
        outlier_score = outlier_num*num_score + np.sum(outlier_dist_array)
        score_list.append(outlier_score)

    is_burst = np.zeros(avail_horizon)
    for i in range(len(num_list)):
        if num_list[i] >=2 and score_list[i] > score_threshold:
            is_burst[i] = 1
    if not status: # non-burst状态
        if np.sum(is_burst)>0:
            return True
        else:
            return False
    else: # burst状态，全部没有才进行切换
        if np.sum(is_burst)==0:
            return False
        else:
            return True

def longterm_result_threshold_validator(longterm_result_total, real_result, seq_len=24):
    """
        longterm_result: (ts,24,3)
        real_result: (ts+seq_len)
    """
    result_list1 = []
    outlier_num_list = []
    outlier_dist = []
    score_list = []
    # result_list2 = []
    # qrisk = QuantileLoss(quantiles=[0.1,0.5,0.9])
    for i in trange(len(longterm_result_total)):
        longterm_result = longterm_result_total[...,1]
        down_thre = longterm_result_total[...,0]
        up_thre = longterm_result_total[...,2]
        real_data = real_result[i:i+seq_len]
        num,dist = calculate_outlier(longterm_result_total[i], real_data)
        outlier_score = num * 0.1 + dist
        result_list1.append(SMAPE(longterm_result[i],real_data))
        outlier_num_list.append(num)
        outlier_dist.append(dist)
        score_list.append(outlier_score)
        # result_list2.append(qrisk.numpy_normalised_quantile_loss(longterm_result_total, real_data, quantile=0.9))
    smape_val = np.mean(result_list1)
    # 判断是否为burst状态
    score_threshold = 1.0 # 0.3过小，检出了2200个点。1.0 检出了1293个点
    is_burst = np.zeros(len(longterm_result_total))
    for i in trange(len(longterm_result_total)):
        if i<23: # 如果数量过小，则直接忽略
            continue
        is_burst[i] = burst_detection(preds=longterm_result_total[i-23:i+1],trues=np.array([real_result[j:j+seq_len] for j in range(i-23,i+1)]),status = is_burst[i-1], score_threshold=score_threshold)
    print(np.sum(is_burst), np.sum(is_burst)/len(is_burst))


def show_prediction_result(preds,real_result,i,seq_len=24):
    real_data = real_result[i:i+seq_len]
    true = real_data
    plt.plot(preds[...,1],color='b')
    plt.plot(true,color='r')
    plt.plot(preds[...,0],color='g')
    plt.plot(preds[...,2],color='g')
    plt.show()

# np.where(is_burst==1)[0]
# i=5092
# show_prediction_result(longterm_result_total[i], real_result, i)