"""
This project use source code from eRL_demo_PPOinSingleFile.py 
from [ElegantRL](https://github.com/AI4Finance-Foundation/ElegantRL), 
copyright Yonv1943曾伊言
licensed under the Apache 2.0 license. Followed by the whole Apache 2.0 license text.
"""
import gym
from basic_env.basic_env import BasicEnv, BasicStackEnv
from basic_env.predict_env import PredictBasicEnv, APPOEnv
from util.constdef import IF_RETRAIN, create_folder_if_not_exist, OUTPUT_DIR, CACHE_DATA_DIR, BURSTY_CACHE_DIR
from tqdm import trange, tqdm
import pandas as pd
import numpy as np

from ppo.common import Arguments, PreprocessEnv, train_and_evaluate, train_burst_scaler
from ppo.agent import AgentDiscretePPO, AgentPPO
from ppo.validator import validate_actor
from util.constdef import TRAIN_MODE, TEST_MODE, VALI_MODE, ACTION_MAX
from ppo.validator import workload_validator
from util.workload_validator import onestep_workload_validator
from test_base import burst_validator
from baseline_method.baseline_burstaware_method import burst_aware_prediction_method, threshold_validator
from baseline_method.baseline_prediction_method import allocate_based_on_onestep, allocate_based_on_longterm
from baseline_method.baseline_asarsa_method import asarsa_validator, asarsa_trainer
from baseline_method.baseline_sarsa_agent import sarsa_asarsa_trainer, sarsa_asarsa_validator
from ppo.sac import AgentModSAC
from baseline_method.dqn.agent import AgentDoubleDQN, AgentDQN
from baseline_method.dqn.run import train_and_evaluate_dqn
from util.constdef import config_instance, TOTAL_STEP
gym.logger.set_level(40)  # Block warning

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

'''demo.py'''
import logging
import os
def logger_register(logger_name: str = '', outputFilename: str = "main_informer_Function.log") -> None:
    """
        设置日志格式，使其同时在文件与命令行中进行输出
        该函数需要被放在最前，先于第一个执行的logging函数
    :param outputFilename: 日志输出的文件名称
    :param logger_name: str 日志记录器的名称
    :return: None
    """
 
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s  %(filename)s:%(lineno)d : %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %A %H:%M:%S'
    )

    import pathlib
    # pathlib.Path(os.path.join('output','log')).mkdir(parents=True, exist_ok=True) 
    fileHandler = logging.FileHandler(outputFilename)  # 默认mode=a
    fileHandler.setLevel(logging.DEBUG)
    fileHandler.setFormatter(formatter)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.INFO)
    consoleHandler.setFormatter(formatter)

    logger.addHandler(fileHandler)
    logger.addHandler(consoleHandler)


def demo_discrete_action(dataStore, workload_name, action_max=1,
    train_step=10000, vali_step=3000, test_step=3000, start_point = 30,
    use_burst=True, is_burst=False,sla=8.,retrain=True, des_mean=10000, des_std=3000,
    use_model=0, use_continuous=False,use_two_step=False, use_env = 0):
    args = Arguments()  # hyper-parameters of on-policy is different from off-policy
    if use_model == 0:
        if not use_continuous:
            args.agent = AgentDiscretePPO() # 使用离散版本的Agent，如果连续动作需要对此项进行修改
        else:
            args.agent = AgentPPO()
            args.agent.if_use_cri_target = True
    elif use_model == 1:
        args.agent = AgentModSAC()
    elif use_model == 2:
        args.agent = AgentDQN()
    args.visible_gpu = '1'
    # args.env = PreprocessEnv(env=APPOEnv(workload_name = workload_name, action_max=action_max)) # 传统做法
    if use_env == 0:
        args.env = PreprocessEnv(env=PredictBasicEnv(workload_name = workload_name, action_max=action_max,
            train_step=train_step, vali_step=vali_step, test_step=test_step, start_point=start_point,
            use_burst=use_burst,sla=sla, des_mean=des_mean, des_std=des_std,use_continuous=use_continuous))
    elif use_env == 1:
        args.env = PreprocessEnv(env=APPOEnv(workload_name = workload_name, action_max=action_max,
            train_step=train_step, vali_step=vali_step, test_step=test_step, start_point=start_point,
            use_burst=use_burst,sla=sla, des_mean=des_mean, des_std=des_std))
    if use_continuous:
        args.env.if_discrete = False
    else:
        args.env.if_discrete = True
    args.env.target_return = 50000 # 目标期望，达到后训练停止
    # DEBUG 调整最长步数
    args.break_step = TOTAL_STEP
    args.target_step = args.env.max_step 
    args.if_per_or_gae = True # DEBUG 暂时保持GAE开启
    args.gamma = 0.5
    args.workload_name = workload_name
    args.init_before_training(if_main=True)
    if (use_burst and not is_burst) or not use_burst: # non-burst或者不用burst
        if use_model == 2:
            train_and_evaluate_dqn(args, use_burst=use_burst,use_two_step=use_two_step)
        else:
            train_and_evaluate(args,use_burst=use_burst,use_two_step=use_two_step)
    else:
        # DEBUG 默认认为所有burst-aware的部分都需要重新训练
        retrain = True
        dataStore = train_burst_scaler(dataStore, args,cur_mode=TRAIN_MODE,force_retrain=retrain)
        dataStore = train_burst_scaler(dataStore, args,cur_mode=VALI_MODE,force_retrain=retrain)
        dataStore = train_burst_scaler(dataStore, args,cur_mode=TEST_MODE,force_retrain=retrain)
    return dataStore

def validate_discrete_action(dataStore, workload_name,cur_mode=TRAIN_MODE,action_max=1,
    train_step=10000, vali_step=3000, test_step=3000, start_point=30,
    in_burst=False,train_burst=0,sla=8., des_mean=10000, des_std=3000, use_model=0,use_env=0,use_continuous=False):
    """
    in_burst表示读取的模型来自哪个文件夹(True时表示用non-burst训练，False时表示使用原来的模型)
    use_burst表示是否使用burst训练的模型
    train_burst表示是否使用burst去训练，0表示不用，1表示使用non-burst数据，2表示使用burst数据
    """
    args = Arguments()  # hyper-parameters of on-policy is different from off-policy
    if use_model == 0:
        if not use_continuous:
            args.agent = AgentDiscretePPO() # 使用离散版本的Agent，如果连续动作需要对此项进行修改
        else:
            args.agent = AgentPPO()
            args.agent.if_use_cri_target = True
    elif use_model == 1:
        args.agent = AgentModSAC()
    elif use_model == 2:
        args.agent = AgentDQN()
    args.visible_gpu = '1'
    # 传统
    # args.env = PreprocessEnv(env=APPOEnv(workload_name = workload_name, action_max=action_max))
    remark = ""
    if train_burst == 0: # 使用全体数据测试
        use_burst = False
        if use_env == 0:
            args.env = PreprocessEnv(env=PredictBasicEnv(workload_name = workload_name, action_max=action_max, 
                train_step=train_step, vali_step=vali_step, test_step=test_step, start_point=start_point,
                use_burst=use_burst,sla=sla, des_mean=des_mean, des_std=des_std,use_continuous=use_continuous))
        elif use_env == 1:
            args.env = PreprocessEnv(env=APPOEnv(workload_name = workload_name, action_max=action_max, 
                train_step=train_step, vali_step=vali_step, test_step=test_step, start_point=start_point,
                use_burst=use_burst,sla=sla, des_mean=des_mean, des_std=des_std))
        remark = "non-burst-aware"
    elif train_burst == 1: # 使用non-burst数据测试
        use_burst = True
        if use_env == 0:
            args.env = PreprocessEnv(env=PredictBasicEnv(workload_name = workload_name, action_max=action_max, 
                train_step=train_step, vali_step=vali_step, test_step=test_step, start_point=start_point,
                use_burst=use_burst,sla=sla, des_mean=des_mean, des_std=des_std,use_continuous=use_continuous))
        elif use_env == 1:
            args.env = PreprocessEnv(env=APPOEnv(workload_name = workload_name, action_max=action_max, 
                train_step=train_step, vali_step=vali_step, test_step=test_step, start_point=start_point,
                use_burst=use_burst,sla=sla, des_mean=des_mean, des_std=des_std))
        remark = "burst-aware/non-burst"
    elif train_burst == 2: # 使用burst数据测试
        use_burst = True
        if use_env == 0:
            args.env = PreprocessEnv(env=PredictBasicEnv(workload_name = workload_name, action_max=action_max, 
                train_step=train_step, vali_step=vali_step, test_step=test_step, start_point=start_point,
                use_burst=use_burst,sla=sla,is_burst_bool=True, des_mean=des_mean, des_std=des_std,use_continuous=use_continuous))
        elif use_env == 1:
            args.env = PreprocessEnv(env=APPOEnv(workload_name = workload_name, action_max=action_max, 
                train_step=train_step, vali_step=vali_step, test_step=test_step, start_point=start_point,
                use_burst=use_burst,sla=sla,is_burst_bool=True, des_mean=des_mean, des_std=des_std))
        remark = "burst-aware/burst"

    if use_continuous:
        args.env.if_discrete = False
    else:
        args.env.if_discrete = True
    args.env.target_return = 50000 # 目标期望，达到后训练停止
    args.target_step = args.env.max_step 
    args.if_per_or_gae = True
    args.gamma = 0.5
    args.init_before_training(if_main=True)
    dataStore, _, _ = validate_actor(dataStore, args=args, workload_name=workload_name,cur_mode=cur_mode, res_sla = sla,
        use_burst=use_burst,in_burst=in_burst,train_burst=train_burst,remark=remark) # 使用基础动作和
    return dataStore

def validator_prediction_result(workload_name):
    workload_validator(workload_name)

def burst_aware(dataStore,workload_name,action_max=5,sla=8., 
    train_step=10000, vali_step=3000, test_step=3000, start_point = 30,
    des_mean=10000, des_std=3000,is_validate_result=False,train_burst=True,use_model=0, use_continuous=False,use_two_step=False):
    action_max = action_max
    use_burst = True
    # NOTE non_burst训练
    if not train_burst:
        if not is_validate_result:
            logging.info("train burst-aware/non-burst ppo agent")
            _ = demo_discrete_action(dataStore,workload_name=workload_name, action_max=action_max,
                train_step=train_step, vali_step=vali_step, test_step=test_step, start_point = start_point,
                use_burst=True,is_burst=False,sla=sla, des_mean=des_mean, des_std=des_std, use_model=use_model, retrain=IF_RETRAIN, use_continuous=use_continuous,use_two_step=use_two_step)
        else:
        # NOTE non-burst 验证数据
            logging.info("validate burst-aware/non-burst ppo agent")
            is_burst = 1
            cur_mode_list={TRAIN_MODE,VALI_MODE,TEST_MODE}
            in_burst = True
            if train_choice == 3: # 对于PPO训练的第三种情况，在验证的时候需要使用non-burst的数据在burst-aware的情况下训练
                in_burst = False
            for cur_mode in cur_mode_list:
                dataStore = validate_discrete_action(dataStore,workload_name=workload_name,
                    cur_mode=cur_mode,action_max=action_max,
                    train_step=train_step, vali_step=vali_step, test_step=test_step, start_point = start_point,
                    train_burst=is_burst,sla=sla,in_burst=in_burst,
                    des_mean=des_mean, des_std=des_std, use_model=use_model, use_continuous=use_continuous)
    # NOTE burst部分训练与验证数据
    else:
        dataStore = demo_discrete_action(dataStore,workload_name=workload_name, action_max=action_max,
            train_step=train_step, vali_step=vali_step, test_step=test_step, start_point = start_point,
            use_burst=True,is_burst=True,sla=sla, des_mean=des_mean, des_std=des_std, use_model=use_model, retrain=IF_RETRAIN, use_continuous=use_continuous,use_two_step=use_two_step)

    # 对比实验，调整train_burst参数，0表示不用，1表示Non-burst，2表示burst
    # is_burst = 1
    # validate_discrete_action(workload_name=workload_name,cur_mode=TEST_MODE,action_max=action_max,use_burst=True,in_burst=True,train_burst=is_burst,sla=sla, des_mean=des_mean, des_std=des_std)
    return dataStore

def init_file(csv_file_name):
    create_folder_if_not_exist(OUTPUT_DIR)
    create_folder_if_not_exist(CACHE_DATA_DIR)
    create_folder_if_not_exist(BURSTY_CACHE_DIR)

def output2file(dataStore, csv_file_name):
    dataStore.to_csv(os.path.join(OUTPUT_DIR, csv_file_name), index=False,mode='w')

def run_ppo_burst_aware_agent(data_list, dataStore, sla=8., action_max=5,
    is_burst_aware=False,train_burst=False,is_validate_result=False,
    train_step = 10000, vali_step = 3000, test_step=3000, start_point = 30,
    des_mean=10000, des_std=3000, use_model = 0, 
    use_continuous=False, use_two_step=False):
    """
    Args:
        * data_list，表示遍历的数据集合
        * dataStore，运行结果记录
        * is_burst_aware: 是否对burst开启检测。True则开启，False不开启
        * is_validate_result: 是否开启检验，True则开启，False不开启
    """
    for workload_name in data_list:
        logging.info(f"ppo using {workload_name} as workload")
        use_burst = False
        # NOTE 训练部分（不使用burst，原版对比）
        if not is_burst_aware:
            if not is_validate_result:
                logging.info("train burst-aware ppo-agent")
                _ = demo_discrete_action(dataStore,workload_name=workload_name, action_max=action_max,
                    train_step=train_step, vali_step=vali_step, test_step=test_step, start_point = start_point,
                    use_burst=use_burst,sla=sla, des_mean=des_mean, des_std=des_std, use_model = use_model, retrain=IF_RETRAIN, use_continuous=use_continuous,use_two_step=use_two_step)
            # NOTE non-burst-aware验证
            else:
                logging.info("validate burst-aware ppo-agent")
                is_burst = 0
                cur_mode_list={TRAIN_MODE,VALI_MODE,TEST_MODE}
                in_burst = False
                if train_choice == 4: # 对于第四种情况，尽管是在全局上进行训练，但是验证使用的是burst-aware的模型
                    in_burst = True
                for cur_mode in cur_mode_list:
                    dataStore = validate_discrete_action(dataStore,
                        workload_name=workload_name,cur_mode=cur_mode,
                        train_step=train_step, vali_step=vali_step, test_step=test_step, start_point = start_point,
                        action_max=action_max,train_burst=is_burst,sla=sla,in_burst=in_burst,
                        des_mean=des_mean, des_std=des_std, use_model = use_model, use_continuous=use_continuous)
        # NOTE burst相关测试
        else:
            logging.info("train and validate burst-aware ppo-agent")
            dataStore = burst_aware(dataStore,workload_name,action_max=action_max,sla=sla,
                train_step=train_step, vali_step=vali_step, test_step=test_step, start_point = start_point,
                des_mean=des_mean, des_std=des_std,
                is_validate_result=is_validate_result,train_burst=train_burst, use_model = use_model, use_continuous=use_continuous,use_two_step=use_two_step)

        # 验证，纯粹输出流量的预测效果图
        # validator_prediction_result(workload_name)
        # 测试，训练burst部分单步预测器的效果
        # burst_validator(workload_name = workload_name,use_burst=use_burst)
    return dataStore

if __name__ == '__main__':
    global train_choice
    logger_register()
    # validate会输出的对应文件
    aim_csv_file = "validate_output.csv" # 最终验证结果会在对应的未知输出
    init_file(aim_csv_file)
    dataStore = pd.DataFrame(
        columns=[
            "remark",
            "setting", # 与validate结果保存相关的setting
            "workload_name",# 流量名称
            "env_name",     # 使用的环境名称（方法）
            "cur_mode",     # 当前所属的阶段
            "total_reward", # 强化学习的奖励值。如果不涉及到强化学习相关逻辑则设置为0
            "sla",          # sla违约总数
            "sla_total",    # SLA违约超出数（总计）
            "instance",     # 使用的总实例数
            "avg_utilization", # 平均CPU占用率
            "optimal_instance", # 理想实例数
            "over_instance", # 超供量
            "under_instance", # 不足量
            "steps",        # 结果中的总步长数
            "onestep_predict", # 使用单步预测的比例
            "longterm_predict", # 使用多步预测的比例
            "addition_info1", # 额外槽位1
            "addition_info2", # 额外槽位2
            ])
    # 记录信息的方式 dataStore = dataStore.append({"setting":xxx},ignore_index=True)这样就可以完成记录
    # NOTE 超参数部分
    des_mean=500
    des_std=175
    sla=16.
    action_max=10
    assert action_max == ACTION_MAX
    # train_step = 10000
    # vali_step = 3000
    # test_step = 3000
    start_point = 30
    train_step = 6000
    vali_step = 2000
    test_step = 2000


    data_list = ["2018_FIFA_World_Cup","Barack_Obama","Donald_Trump","Elizabeth_II","Elon_Musk","Facebook","Game_of_Thrones","Google","United_States","YouTube"]
    # data_list = ["2018_FIFA_World_Cup","Google"] # real world

    # reward_ratio_change_list = [0.1,0.3,0.5,0.7,0.9]
    reward_ratio_change_list = [0.1]

    # 强化学习功能
    is_validate_result = False # 是否开启验证
    # 下列选择一个
    choose_use_ppo = False # PPO主方法
    choose_use_burst_aware_prediction = False # 使用burst-aware
    choose_use_asarsa = False # A-SARSA
    choose_use_onestep_prediction = False# 使用纯单步预测
    choose_use_multistep_prediction = False# 使用纯多步预测
    choose_use_sarsa_asarsa = False # SARSA版本的ASARSA模型
    choose_use_reactive_threshold = False # 简单的阈值法

    validate_current_onestep_prediction_result = False # 验证单步预测结果
    # 选择使用哪一个
    choose_use_ppo = True
    # choose_use_reactive_threshold = True
    # =================================================================================================
    # 选择burst模式
    train_choice = 1 # 选择
    # ! 注意：burst-aware的时候，必须跑完burst部分(train_choice==2)，再跑non-burst部分(train_choice==1)
    is_burst_aware = False
    train_burst = False
    # non-burst-aware 0
    if train_choice == 0:
        is_burst_aware = False
    # burst-aware/non-burst 1
    elif train_choice == 1:
        is_burst_aware = True
        train_burst = False
    # burst-aware/burst 2
    elif train_choice == 2:
        is_burst_aware = True
        train_burst = True
    elif train_choice == 3:
        # 训练时使用全体，验证时使用non-burst
        # 目标：探索新的burst-aware/non-burst训练模式
        # * 实际证明，效果一般
        if not is_validate_result: # 训练
            is_burst_aware = False
        else:
            is_burst_aware = True
            train_burst = False
    elif train_choice == 4:
        # 训练时使用non-burst，但是在验证时使用全体数据
        # 目标：验证新的burst-aware/non-burst + burst的总体效果
        if not is_validate_result:
            is_burst_aware = True
            train_burst = False
        else:
            is_burst_aware = False
    else:
        raise NotImplementedError
    # 以下为PPO
    # =============================================================================================================================================
    if choose_use_ppo:
        # NOTE 选择分支部分
        use_model = 0 # 0 PPO, 1 SAC, 2 DQN
        # 如无意外，禁止开启use_continuous，该选项会覆盖掉对应的位置
        use_continuous = False # 是否采用连续动作，仅对PPO有效。目前测试的结果是连续动作并没有太大差别
        # 开启两步式训练
        use_two_step = False # 是否开启两步训练，即先训练好基础，再加上addon

        for cur_reward_ratio in reward_ratio_change_list:
            logging.info(f"ppo reward ratio{cur_reward_ratio}")
            config_instance.reward_bias_ratio = cur_reward_ratio
            dataStore = run_ppo_burst_aware_agent(data_list,dataStore,
                train_step=train_step, vali_step = vali_step, test_step=test_step, start_point = start_point,
                sla=sla,action_max=action_max, des_mean=des_mean, des_std=des_std,
                is_burst_aware=is_burst_aware,train_burst=train_burst,is_validate_result=is_validate_result,
                use_model=use_model, use_continuous=use_continuous, use_two_step=use_two_step)
            if train_choice == 2:
                # burst-aware与reward_ratio无关，只需要计算一次
                break
    # =============================================================================================================================================
    # PPO部分结束
    # burst-aware对比算法
    if choose_use_burst_aware_prediction:
        for workload_name in data_list:
            logging.info(f"baseline/burst-aware {workload_name}")
            dataStore = burst_aware_prediction_method(dataStore,workload_name=workload_name,
                train_step=train_step, vali_step = vali_step, test_step=test_step,
                sla=sla, des_mean = des_mean, des_std=des_std)
    # ASARSA算法
    if choose_use_asarsa:
        # ! 注意：burst-aware的时候，必须跑完burst部分，再跑non-burst部分
        use_model = 2 # 0为PPO，2为DQN
        is_burst_aware = False
        for cur_reward_ratio in reward_ratio_change_list:
            config_instance.reward_bias_ratio = cur_reward_ratio
            for workload_name in data_list:
                if is_validate_result:
                    if use_model == 2:
                        dataStore = asarsa_validator(dataStore, workload_name,
                            train_step=train_step, vali_step=vali_step, test_step=test_step, 
                            action_max=action_max, sla=sla, des_mean=des_mean, des_std=des_std,
                            use_burst=is_burst_aware, is_burst=train_burst)
                    else:
                        for cur_mode in [TRAIN_MODE, VALI_MODE, TEST_MODE]:
                            dataStore = validate_discrete_action(dataStore,
                                workload_name=workload_name,cur_mode=cur_mode,
                                train_step=train_step, vali_step=vali_step, test_step=test_step, start_point = start_point,
                                action_max=action_max,sla=sla,
                                des_mean=des_mean, des_std=des_std, use_model = use_model, use_env=1)    
                else:
                    if use_model == 2:
                        dataStore = asarsa_trainer(dataStore, workload_name,
                            train_step=train_step, vali_step=vali_step, test_step=test_step, 
                            action_max=action_max, sla=sla, des_mean=des_mean, des_std=des_std,
                            use_burst=is_burst_aware, is_burst=train_burst)
                    else:
                        dataStore = demo_discrete_action(dataStore,workload_name=workload_name, action_max=action_max,
                            train_step=train_step, vali_step=vali_step, test_step=test_step, start_point = start_point,
                            use_burst=False, sla=sla, des_mean=des_mean, des_std=des_std, use_model=use_model, use_env=1)
    # 执行单步预测结果
    if choose_use_onestep_prediction:
        logging.info("开始执行基于单步预测的资源分配")
        for workload_name in data_list:
            logging.info(f"开始执行基于单步预测的资源分配，流量{workload_name}")
            dataStore = allocate_based_on_onestep(dataStore,workload_name=workload_name,
                train_step=train_step, vali_step = vali_step, test_step=test_step,
                sla=sla, des_mean = des_mean, des_std=des_std)
    if choose_use_multistep_prediction:
        logging.info("开始执行基于长期预测的资源分配")
        for workload_name in data_list:
            logging.info(f"开始执行基于长期预测的资源分配，流量{workload_name}")
            dataStore = allocate_based_on_longterm(dataStore,workload_name=workload_name,
                train_step=train_step, vali_step = vali_step, test_step=test_step,
                sla=sla, des_mean = des_mean, des_std=des_std)
    # SARSA版本
    # ASARSA原版
    if choose_use_sarsa_asarsa:
        logging.info("开始执行ASARSA(SARSA版本)")
        is_burst_aware = False # 强制使用non-burst

        for workload_name in data_list:
            if not is_validate_result:
                logging.info(f"训练ASARSA(SARSA版本),流量{workload_name}")
                sarsa_asarsa_trainer(workload_name = workload_name,
                        train_step=train_step, vali_step=vali_step, test_step=test_step, 
                        action_max=action_max, sla=sla, des_mean=des_mean, des_std=des_std,
                        use_burst=is_burst_aware, is_burst=train_burst)
            else:
                logging.info(f"验证ASARSA(SARSA版本),流量{workload_name}")
                dataStore = sarsa_asarsa_validator(dataStore, workload_name = workload_name,
                        train_step=train_step, vali_step=vali_step, test_step=test_step, 
                        action_max=action_max, sla=sla, des_mean=des_mean, des_std=des_std,
                        use_burst=is_burst_aware, is_burst=train_burst)
    if validate_current_onestep_prediction_result:
        logging.info("开始验证单步预测的结果（by SMAPE）")
        for workload_name in data_list:
            logging.info(f"验证{workload_name}")
            dataStore = onestep_workload_validator(dataStore, workload_name = workload_name,
                        train_step=train_step, vali_step=vali_step, test_step=test_step, 
                        action_max=action_max, sla=sla, des_mean=des_mean, des_std=des_std,
                        use_burst=is_burst_aware, is_burst=train_burst)
    
    if choose_use_reactive_threshold:
        logging.info("启动阈值法算法")
        for workload_name in data_list:
            dataStore = threshold_validator(dataStore,workload_name=workload_name,
                train_step=train_step, vali_step = vali_step, test_step=test_step,
                sla=sla, des_mean = des_mean, des_std=des_std)
    # 将结果缓存
    output2file(dataStore,csv_file_name=aim_csv_file)
    

