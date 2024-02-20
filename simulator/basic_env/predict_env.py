"""
    将预测器作为动作空间的模拟环境，与A-SARSA类模拟环境
"""
import numpy as np
from numpy.core.fromnumeric import repeat
from numpy.core.shape_base import stack
import pandas as pd
import math

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from tqdm import trange, tqdm
import os
from joblib import load

from basic_env.reward_func import origin_reward_for_basic_env, custom_reward, custom_reward_penalty
from util.norm import local_scale
from util.time_features import time_features
from util.constdef import IF_RETRAIN, MAX_WORKLOAD_FACTOR, create_folder_if_not_exist, read_workload_data, get_cpu_predictor, get_res_predictor, read_longterm_prediction_result, parse_workload_data,\
        CACHE_DATA_DIR, TRAIN_MODE, TEST_MODE, VALI_MODE, MAX_RES, MAX_CPU, MAX_WKL, MAX_INS, LONGTERM_EXACT_DIR, LONGTERM_START_POINT
from ppo.validator import get_onestep_validate_result, get_longterm_validate_result
from predictor.performance_predictor import CpuPredictor, ResPredictor
from predictor.longterm_predictor import LongtermPredictor
from predictor.onestep_predictor import OneStepPredictor
from basic_env.base_env import BaseEnv
from basic_env.burst_detector import BurstDetector
from util.constdef import config_instance
import logging

# * burst_detector.py 中，正在进行
class PredictBasicEnv(BaseEnv):
    """
        本文的重点代码，将对预测器结果的置信纳入到动作之中。
        特点就是调度器可以选择是否相信流量预测器的结果
        
        状态空间为：
        stack状态：流量向量
        其他状态：CPU占用率、响应时间、time of features(4个)、单步预测器精度SMAPE的平均值、多步预测器的单步精度90-Risk的平均值
            流量使用本地正则化方法
            精度应该没办法用本地正则化方法，可以考虑除以100
        
        动作空间为：-v/0/+v，是否直接采用预测结果
        存在最小/最大状态设定
        奖励函数由外源指定
    """
    metadata = {'render.modes': ['human']}
    env_name = 'predict_env_basic' # 该环境为基础云计算模型，状态空间为最简单的版本

    def __init__(self, workload_name = 'Norway',
                     start_point = 30, train_step = 10000, test_step = 3000, vali_step = 3000,
                     des_mean = 10000, des_std = 3000,
                     action_max = 2, stack_point_num = 24, longterm_pred_num = 24, sla=8.,
                     use_prediction_instead = False, use_longterm_prediction=True, use_burst=True, is_burst_bool=False, use_continuous=False):
        """
            从指定的流量文件中读取数据，注册到环境中。
            注册奖励函数、状态空间和动作空间
            如果启动了burst_aware的话，这个环境只能提供non-burst的数据。
        Args:
            workload_name: 读取的流量的名称
            train_step: 训练段的长度，需要足够的长来保证轨迹的多样性，但是不能够过长
            des_mean, des_std：训练部分流量经过放缩之后对应的均值和方差
            action_max: 最大的调度实例数量v，动作空间为-v~v
            stack_point_num: 状态空间中嵌套的流量点数，状态空间为 （流量向量，预测精度）*stack_point_num+时间特征(4)+CPU占用+响应时间
            use_prediction_instead：是否使用A-SARSA类似的方法，用预测的结果直接算出未来的响应时间和CPU占用率，用这个结果来指导调度。
            use_burst: 是否开启使用burst
            is_burst_bool: True为只使用burst部分数据，Fals表示只使用non-burst部分数据
        """
        assert type(action_max) == type(1)
        assert type(stack_point_num) == type(1)

        self.workload_name = workload_name
        self.use_continuous = use_continuous
        self.pred_action_max = action_max # * 最近的尝试，将基准点探索进行剥离s
        # 注册空间，从目前的感觉
        if use_longterm_prediction:
            # self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(stack_point_num+8,), dtype=np.float32) 
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(stack_point_num + 4 + 4,), dtype=np.float32) 
            self.action_space = spaces.Discrete((2*action_max+1) + 2*(2*self.pred_action_max+1))
        else:
            # self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(stack_point_num+7,), dtype=np.float32) 
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32) 
            self.action_space = spaces.Discrete(2*(2*action_max+1))
        self.stack_point_num = stack_point_num
        self.action_max = action_max
        self.sla = sla

        # 注册资源预测器（流量->CPU占用，流量->响应时间）
        #   使用两个SVM预测器完成预测
        # 快速计算缓存，用于单步内流量计算的快速缓存
        self.cpu_cache = {}
        self.res_cache = {}
        #   NOTE 与流量的性质相关，这个地方可能需要修改
        self.MAX_WKL = MAX_WKL # 最大流量250QPS
        self.MAX_CPU = MAX_CPU # 最大CPU占用控制在100%
        self.MAX_RES = MAX_RES # 最大响应时间控制在80ms
        self.MAX_INS = MAX_INS
        self.cpu_predictor = CpuPredictor() # get_cpu_predictor()
        self.res_predictor = ResPredictor()
        self.max_workload_count = self.__calculate_max_reward_workload()
        # 注册流量读取器
        #   从指定csv文件中读取流量信息
        origin_workload_data = read_workload_data(workload_name)
        self.origin_workload_data = origin_workload_data
        # self.longterm_exact_predictor = LongtermPredictor(workload_name,workload_dir=LONGTERM_EXACT_DIR) # P80数据
        self.longterm_predictor = LongtermPredictor(workload_name)
        self.onestep_predictor = OneStepPredictor(workload_name)
        # origin_data, origin_timestamp, origin, pred, true = read_longterm_prediction_result(workload_name)
        origin_timestamp = self.longterm_predictor.origin_timestamp
        longterm_start_point = LONGTERM_START_POINT # 由长期训练部分的len1+len2决定，如果有修改的话需要跨库进行对应的更正（
        self.start_point = start_point  # start_point 标识开始点
        self.train_step = train_step
        self.vali_step = vali_step
        self.test_step = test_step
        self.max_step = max(train_step,vali_step,test_step)
        longterm_cur_start_point = longterm_start_point + self.start_point
        self.longterm_cur_start_point = longterm_cur_start_point
        assert longterm_cur_start_point + self.train_step + self.vali_step + self.test_step <= len(origin_workload_data), "长度不会超标"
        assert start_point + self.train_step + self.vali_step + self.test_step <= len(origin_timestamp), "长度不会超标"
        assert start_point - stack_point_num + 1 >= 0,"开始点需要足够大"
        # 预测精度评估，必须在正则化之前
        self.train_workload_data = origin_workload_data[longterm_cur_start_point-(stack_point_num-1):longterm_cur_start_point+self.train_step] 
        self.test_workload_data = origin_workload_data[longterm_cur_start_point+self.train_step-(stack_point_num-1):longterm_cur_start_point+self.train_step+self.vali_step+self.test_step]
        onestep_accuracy = get_onestep_validate_result(self.onestep_predictor.pred[start_point-stack_point_num+1:start_point+train_step+vali_step+test_step],
                                                        origin_workload_data[longterm_cur_start_point-(stack_point_num-1):longterm_cur_start_point+self.train_step+self.vali_step+self.test_step]) 
        longterm_accuracy = get_longterm_validate_result(self.longterm_predictor.pred[start_point-stack_point_num+1:start_point+train_step+vali_step+test_step],
                                                        origin_workload_data[longterm_cur_start_point-(stack_point_num-1):longterm_cur_start_point+self.train_step+self.vali_step+self.test_step])
        # ? 可以考虑在这个点进行对比，使用正则化之后的数据完成对应的比较
        # 输入的长期预测数据：train_)longterm_prediction，然后重新计算SMAPE
        # 输入的真实数据：train_workload，test_workload
        # * 注意对齐数据（已经检验过，开始点对齐，两边无异常）
        # * 注意train和test部分的数据长度检验
        # self.train_burst_detector = BurstDetector(preds = XXX, trues = XXX)，其中这个XXX应该参考前面计算

        self.use_burst = use_burst
        self.is_burst_bool = is_burst_bool
        longterm_pred_num = self.longterm_predictor.pred.shape[1]
        self.burst_detector = BurstDetector(preds = self.longterm_predictor.pred[start_point-longterm_pred_num+1:start_point+train_step+vali_step+test_step],
                                            real_data = origin_workload_data[longterm_cur_start_point-longterm_pred_num+1:longterm_cur_start_point-1+self.train_step+self.vali_step+self.test_step+longterm_pred_num],
                                            one_step_pred = self.onestep_predictor.pred[start_point-longterm_pred_num+1:start_point-1+train_step+vali_step+test_step+longterm_pred_num],
                                            workload_name=workload_name, is_retrain=IF_RETRAIN,
                                            train_step=train_step, vali_step=vali_step, test_step=test_step,max_reward_count=self.max_workload_count)
        if use_burst:
            # validator 辅助代码：可视化burst区间
# import matplotlib.pyplot as plt
# total_burst_array, _ = self.burst_detector.generate_burst_point_list(self.burst_detector.burst_array)
# true_data = origin_workload_data[longterm_cur_start_point-longterm_pred_num+1+46:longterm_cur_start_point-1+self.train_step+self.vali_step+self.test_step+longterm_pred_num]
# plt.plot(true_data,color='b')
# for s,e in total_burst_array:
#     plt.axvspan(s,e,alpha=0.5,color='red')
# plt.show()
            if not is_burst_bool:
                self.current_point = self.burst_detector.get_next_non_burst()
            else:
                self.current_point = self.burst_detector.get_next_burst()
        else:
            self.current_point = 0  # current_point 标识当前点
        # ! burst_detector的数据不需要做正则化，但是其数据也不能作为参考

        self.train_onestep_accuracy = onestep_accuracy[:stack_point_num-1+train_step]
        self.test_onestep_accuracy = onestep_accuracy[-test_step-vali_step-stack_point_num+1:]

        # 去除掉outlier
        onestep_mean = np.mean(self.train_onestep_accuracy)
        onestep_std = np.std(self.train_onestep_accuracy)
        self.onestep_acc_min = max(onestep_mean-3*onestep_std,np.min(self.train_onestep_accuracy))
        self.onestep_acc_max = min(onestep_mean+3*onestep_std,np.max(self.train_onestep_accuracy))

        self.train_longterm_accuracy = longterm_accuracy[:stack_point_num-1+train_step]
        self.test_longterm_accuracy = longterm_accuracy[-test_step-vali_step-stack_point_num+1:]
        longterm_mean = np.mean(self.train_longterm_accuracy)
        longterm_std = np.std(self.train_longterm_accuracy)
        self.longterm_acc_min = max(longterm_mean-3*longterm_std,np.min(self.train_longterm_accuracy))
        self.longterm_acc_max = min(longterm_mean+3*longterm_std,np.max(self.train_longterm_accuracy))
        logging.info(f"\tenvironment state: longterm start at{start_point-stack_point_num+1}, workload start at {longterm_cur_start_point-(stack_point_num-1)}")
        self._normalized_data(des_mean, des_std) # 规定范围后执行正则化，位置不要随便动
        # 时间戳信息

        self.train_timestamp = origin_timestamp[start_point:start_point+train_step]
        self.test_timestamp = origin_timestamp[start_point+train_step:start_point+train_step+vali_step+test_step]
        self.train_time_features = time_features(self.train_timestamp)
        self.test_time_features = time_features(self.test_timestamp)
        # 预测信息，必须在正则化之后
        # ! 预测数据向右移动一个，以提供预测的结果。只有预测值需要平移，这样会方便一些，唯一的问题是总数会比原来小1
        self.train_longterm_prediction = self.longterm_predictor.pred[start_point+1:start_point+train_step+1]
        self.test_longterm_prediction = self.longterm_predictor.pred[start_point+train_step+1:start_point+train_step+vali_step+test_step+1]
        self.train_onestep_prediction = self.onestep_predictor.pred[start_point+1:start_point+train_step+1]
        self.test_onestep_prediction = self.onestep_predictor.pred[start_point+train_step+1:start_point+train_step+vali_step+test_step+1]

        # 其他状态
        self.cur_mode = TRAIN_MODE # 0为训练模式，1为测试模式
        self.instance = 1

        self.allow_bias = 0.05 # 允许在搜索最佳per_worklaod的时候有5%的误差存在
        self.min_bias = 0.2 # 在搜索的时候，最大搜索范围不会超过0.2
        self.use_prediction_instead = use_prediction_instead # 使用A-SARSA类，使用预测值估计响应时间
        self.use_longterm_prediction = use_longterm_prediction # 使用下一个点的长期预测结果
        # * error_ratio 考虑中的随机值，此状态出现会直接改变响应时间（极大提升），可以用作噪声添加与干扰

    def step(self, action):
        """
            输入动作并执行。
            action 为 2 * action_max + 2，其中最后一个是目标值
            返回state, reward, done, info
        """
        if self.use_continuous:
            action = np.where(np.max(action) == action)[0][0]
        illegal_reward_penalty = 0
        if action < 2*self.action_max+1:
            self.instance += action - self.action_max
        elif action < (2*self.action_max+1) + (2*self.pred_action_max+1): # 使用单步预测结果
            delta_instance = action - (2*self.action_max+1) - self.pred_action_max
            predict_workload = self._get_latest_onestep_prediction_result(self.current_point)
            # ! 是否选择进行误差分析，我觉得不应该进行误差分析。
            # predict_bias =  self._get_latest_onestep_bias_by_point(self.current_point,horizon=168)
            # predict_workload += predict_bias
            self.instance = self._judge_instance_num(predict_workload) + delta_instance
        else: # 使用多步预测的上阈值
            delta_instance = action - (2*self.action_max+1) - (2*self.pred_action_max+1) - self.pred_action_max
            predict_workload = self._get_latest_longterm_prediction_result(self.current_point)[-1]
            self.instance = self._judge_instance_num(predict_workload) + delta_instance
        if self.instance <= 0:
            self.instance = 1
            illegal_reward_penalty = -3
        elif self.instance > MAX_INS:
            self.instance = MAX_INS
            illegal_reward_penalty = -3


        done = False
        self.current_point += 1
        if self.cur_mode == TRAIN_MODE and self.current_point >= self.train_step - 1:
            done = True
        elif self.cur_mode == VALI_MODE and self.current_point >= self.vali_step - 1:
            done = True
        elif self.cur_mode == TEST_MODE and self.current_point >= self.test_step - 1:
            done = True

        new_state = self._get_state_by_point(self.current_point, self.instance)
        # reward = self.get_reward(self.get_response_time_from_state(new_state),self.get_cpu_utilization_from_state(new_state))
        reward = self.get_reward(self.get_cur_res(),self.get_cur_cpu())

        if self.use_burst: # 根据现有数据判断新状态是否为burst。如果是，则转为激进方法；否则，继续执行该方法
            result = self.burst_detector.is_burst(self.current_point,mode=self.cur_mode)
            if result == 1:
                info = {"is_burst":True}
                done = True # 当前一定是non-burst的。如果由non_burst转为了burst，则
            else:
                info = {"is_burst":False}
        else:
            info = {}
        info.update({"step":self.current_point})
        state = new_state
        reward += illegal_reward_penalty
        return state, reward, done, info # 返回新的状态，从旧状态到新状态的奖励，是否完成，和额外的信息字典

    def reset(self, aim_instance=None,cur_mode = TRAIN_MODE, restart=True):
        """
            重置开始步长
            Args:
                mode: 决定环境处于何种状态。
        """
        # 重置流量信息，得到当前点
        if not self.use_burst:
            self.current_point = 0
        else:
            if restart:
                self.burst_detector.reset()
            if not self.is_burst_bool: # non-burst模式
                self.current_point = self.burst_detector.get_next_non_burst(mode=cur_mode)-1 # 必须要提前一个，这样下一个才会是Non-burst的
                if self.current_point<0: # 如果是第一个，那就没办法了
                    self.current_point=0
            else:
                self.current_point = self.burst_detector.get_next_burst(mode=cur_mode)

        if aim_instance is not None and aim_instance != 0:
            best_ins = aim_instance
        else:
            best_ins = self.calculate_max_reward_instance(self.get_latest_workload_by_point(self.current_point))
        self.instance = max(best_ins,1)
        self.cur_mode = cur_mode
        return self._get_state_by_point(self.current_point, self.instance)
    
    def _get_latest_onestep_bias_by_point(self, aim_point=None, horizon=24):
        """
        计算残差的范围为过去24个点
        """
        if aim_point == None:
            aim_point = self.current_point

        if self.cur_mode == TRAIN_MODE:
            index = self.start_point+1
        elif self.cur_mode == VALI_MODE:
            index = self.start_point + self.train_step + 1
        elif self.cur_mode == TEST_MODE:
            aim_point += self.vali_step
            index = self.start_point + self.train_step + 1
        else:
            raise NotImplementedError

        if index+aim_point-horizon < 0:
            return 0
        # 选择non-burst部分的
        past_pred = self.onestep_predictor.pred[index+aim_point-horizon:index+aim_point]
        # past_true = self.onestep_predictor.true[index+aim_point-horizon:index+aim_point]
        past_true = self.norm_workload_data[self.longterm_cur_start_point+aim_point+1-horizon:self.longterm_cur_start_point+aim_point+1]
        past_loss = past_pred.ravel() - past_true # 损失必须为预测与真值的差
        if self.use_burst:
            for i in range(aim_point - horizon - 1, aim_point-1):
                if i<0:
                    continue
                elif self.burst_detector.burst_array[i] == 1:
                    pivot = i - (aim_point-horizon-1)
                    past_loss[pivot]=0
        past_loss.sort()
        return past_loss[-int(0.3*len(past_loss))]


    def get_latest_workload_by_point(self, aim_point=None):
        if aim_point == None:
            aim_point = self.current_point

        if self.cur_mode == TRAIN_MODE:
            return self.train_workload_data[aim_point+self.stack_point_num-1]
        elif self.cur_mode == VALI_MODE:
            return self.test_workload_data[aim_point+self.stack_point_num-1]
        elif self.cur_mode == TEST_MODE:
            aim_point += self.vali_step
            return self.test_workload_data[aim_point+self.stack_point_num-1]
        else:
            raise NotImplementedError

    def _get_workload_by_point(self, aim_point):
        if self.cur_mode == TRAIN_MODE:
            workload = self.train_workload_data[aim_point:aim_point+self.stack_point_num]
            return local_scale(workload,workload[-1])
        elif self.cur_mode == VALI_MODE:
            workload = self.test_workload_data[aim_point:aim_point+self.stack_point_num]
            return local_scale(workload,workload[-1])
        elif self.cur_mode == TEST_MODE:
            aim_point += self.vali_step
            workload = self.test_workload_data[aim_point:aim_point+self.stack_point_num]
            return local_scale(workload,workload[-1])
        else:
            raise NotImplementedError
    
    def _get_noscale_workload_by_point(self, aim_point):
        if self.cur_mode == TRAIN_MODE:
            workload = self.train_workload_data[aim_point:aim_point+self.stack_point_num]
        elif self.cur_mode == VALI_MODE:
            workload = self.test_workload_data[aim_point:aim_point+self.stack_point_num]
        elif self.cur_mode == TEST_MODE:
            aim_point += self.vali_step
            workload = self.test_workload_data[aim_point:aim_point+self.stack_point_num]
        else:
            raise NotImplementedError
        return workload/(MAX_WORKLOAD_FACTOR * self.des_mean)

    def _get_latest_onestep_prediction_result(self, aim_point):
        # 默认作为SMAPE进行正则化
        # 预测结果不应该做本地正则化，因为大小的相对差异对最终的动作是有影响的
        if self.cur_mode == TRAIN_MODE:
            return self.train_onestep_prediction[aim_point]
        elif self.cur_mode == VALI_MODE:
            return self.test_onestep_prediction[aim_point]
        elif self.cur_mode == TEST_MODE:
            aim_point += self.vali_step
            return self.test_onestep_prediction[aim_point]
        else:
            raise NotImplementedError

    def _get_latest_longterm_prediction_result(self, aim_point):
        # 默认作为SMAPE进行正则化
        # 预测结果不应该做本地正则化，因为大小的相对差异对最终的动作是有影响的
        if self.cur_mode == TRAIN_MODE:
            return self.train_longterm_prediction[aim_point][0]
        elif self.cur_mode == VALI_MODE:
            return self.test_longterm_prediction[aim_point][0]
        elif self.cur_mode == TEST_MODE:
            aim_point += self.vali_step
            return self.test_longterm_prediction[aim_point][0]
        else:
            raise NotImplementedError

    def _get_onestep_prediction_accuracy(self, aim_point):
        # 默认作为SMAPE进行正则化
        # 预测结果不应该做本地正则化，因为大小的相对差异对最终的动作是有影响的
        if self.cur_mode == TRAIN_MODE:
            return self.train_onestep_accuracy[aim_point:aim_point+self.stack_point_num]
        elif self.cur_mode == VALI_MODE:
            return self.test_onestep_accuracy[aim_point:aim_point+self.stack_point_num]
        elif self.cur_mode == TEST_MODE:
            aim_point += self.vali_step
            return self.test_onestep_accuracy[aim_point:aim_point+self.stack_point_num]

    def _get_longterm_prediction_accuracy(self, aim_point):
        # 默认作为SMAPE进行正则化
        # 预测结果不应该做本地正则化，因为大小的相对差异对最终的动作是有影响的
        if self.cur_mode == TRAIN_MODE:
            return self.train_longterm_accuracy[aim_point:aim_point+self.stack_point_num]
        elif self.cur_mode == VALI_MODE:
            return self.test_longterm_accuracy[aim_point:aim_point+self.stack_point_num]
        elif self.cur_mode == TEST_MODE:
            aim_point += self.vali_step
            return self.test_longterm_accuracy[aim_point:aim_point+self.stack_point_num]

    def _get_features(self, aim_point):
        if self.cur_mode == TRAIN_MODE:
            return self.train_time_features[aim_point]
        elif self.cur_mode == VALI_MODE:
            return self.test_time_features[aim_point]
        elif self.cur_mode == TEST_MODE:
            aim_point += self.vali_step
            return self.test_time_features[aim_point]

    def get_cur_cpu(self):
        current_workload = self.get_latest_workload_by_point(self.current_point)
        current_ins = self.instance
        return self.cpu_cache[int(current_workload / current_ins)]

    def get_cur_res(self):
        current_workload = self.get_latest_workload_by_point(self.current_point)
        current_ins = self.instance
        return self.res_cache[int(current_workload / current_ins)]
    
    def get_cpu_state(self,cpu_value):
        """
        输入0~100的cpu_value，将其映射到-1~1
        """
        if cpu_value > self.MAX_CPU:
            cpu_value = self.MAX_CPU
        elif cpu_value < 0:
            cpu_value = 0
        return cpu_value/self.MAX_CPU*2 - 1
    
    def get_res_state(self,res_value):
        """
        输入一个正数的响应函数值，将其映射到-1~1
        """
        if res_value > self.MAX_RES:
            res_value = self.MAX_RES
        elif res_value < 0:
            res_value = 0
        return res_value/self.MAX_RES*2 - 1

    def get_res_warning_state(self,predict_res_value):
        """
        输入一个预测的响应函数值，将其映射到-1~1
        """
        if predict_res_value >= 2 * self.sla:
            return 1
        else:
            val = -np.log(2-predict_res_value/self.sla)
            val *= 2
            if val > 1:
                val = 1
            elif val < -1:
                val = -1
            return val
    
    def get_onestep_acc_state(self,one_step_acc):
        """
        尽量将准确值映射
        """
        # if one_step_acc < self.onestep_acc_min:
        #     one_step_acc = self.onestep_acc_min
        # elif one_step_acc > self.onestep_acc_max:
        #     one_step_acc = self.onestep_acc_max
        acc_val = (one_step_acc - self.onestep_acc_min) / (self.onestep_acc_max - self.onestep_acc_min)
        return acc_val*2 - 1

    def get_longterm_acc_state(self,long_term_acc):
        """
        尽量将准确值映射
        """
        # if long_term_acc < self.longterm_acc_min:
        #     long_term_acc = self.longterm_acc_min
        # elif long_term_acc > self.longterm_acc_max:
        #     long_term_acc = self.longterm_acc_max
        acc_val = (long_term_acc - self.longterm_acc_min) / (self.longterm_acc_max - self.longterm_acc_min)
        return acc_val*2-1

    def get_ins_state(self, instance_val):
        ins_val = instance_val / self.MAX_INS
        return ins_val*2-1

    def _get_state_by_point(self, aim_point, instance):
        """
            给定目标时间点和当时的实例数，返回一个状态
        """
        workload = self._get_noscale_workload_by_point(aim_point)
        # workload = self._get_workload_by_point(aim_point)
        current_features = self._get_features(aim_point)
        if self.use_prediction_instead:
            raise NotImplementedError
            # current_workload = self._get_latest_onestep_prediction_result(aim_point)

        current_workload = self.get_latest_workload_by_point(aim_point)
        current_accuracy = self._get_onestep_prediction_accuracy(aim_point)
        # current_workload_features = self._get_workload_features(aim_point)
        per_workload = np.array([current_workload / instance]).reshape(-1,1)
        if per_workload.item() < self.MAX_WKL:
            cpu_predict_result = self.cpu_predictor.predict_once(per_workload, instance=self.instance, use_sampling=True)
            res_predict_result = self.res_predictor.predict_once(per_workload, instance=self.instance, use_sampling=True)
        else:
            cpu_predict_result = self.MAX_CPU
            res_predict_result = self.MAX_RES
        self.cpu_cache[int(per_workload.item())] = cpu_predict_result
        self.res_cache[int(per_workload.item())] = res_predict_result

        # 进行正则化
        if self.use_longterm_prediction:
            longterm_accuracy = self._get_longterm_prediction_accuracy(aim_point)
            predict_workload = self._get_latest_onestep_prediction_result(aim_point)
            predict_res = self.res_predictor.predict_once(np.array([predict_workload / instance]).reshape(-1,1), instance=self.instance)
            result = np.array([
                        self.get_cpu_state(cpu_predict_result), 
                        self.get_res_state(res_predict_result),
                        self.get_ins_state(self.instance),
                        self.get_res_warning_state(predict_res)])
            result = np.concatenate([result, 
                        workload,
                        current_features,
                        # self.get_onestep_acc_state(current_accuracy),
                        # self.get_longterm_acc_state(longterm_accuracy),
                        ])
        else:
            result = np.array([cpu_predict_result/self.MAX_CPU, res_predict_result/self.MAX_RES,np.mean(current_accuracy)])
            # result = np.concatenate([result,workload,current_features])
        return result
    
    def get_response_time_from_state(self, state):
        return state[1]*self.MAX_RES
    
    def get_cpu_utilization_from_state(self, state):
        return state[0]*self.MAX_CPU

    def get_reward(self,res,cpu):
        return custom_reward_penalty(res, cpu,response_sla=self.sla)

    def _normalized_data(self, des_mean, des_std):
        """
            完成数据的正则化。正则化必须在刨除异常点后进行，以保证大部分的数据是服从分布的
        """
        # 外部排序，计算分位点
        aim_data = np.sort(self.train_workload_data)
        q1, q3 = aim_data[int(len(aim_data)*0.25)], aim_data[int(len(aim_data)*0.75)]
        iqf = q3-q1
        up_thre, down_thre = q3 + 1.5 * iqf, q1 - 1.5 * iqf
        data_after_filter = aim_data[np.logical_and(aim_data<up_thre,aim_data>down_thre)]
        # 数据清洗，算出清洗后的平均值和方差
        train_mean = np.mean(data_after_filter)
        train_std = np.std(data_after_filter)
        self.des_mean = des_mean
        # train_mean, train_std = lof_smooth_data(self.train_workload_data) 
        self.norm_workload_data = parse_workload_data(self.origin_workload_data, 
                                                        aim_mean = train_mean, aim_std = train_std,
                                                        des_mean = des_mean, des_std = des_std)
        self.train_workload_data = parse_workload_data(self.train_workload_data, 
                                                        aim_mean = train_mean, aim_std = train_std,
                                                        des_mean = des_mean, des_std = des_std)
        self.test_workload_data = parse_workload_data(self.test_workload_data, 
                                                        aim_mean = train_mean, aim_std = train_std,
                                                        des_mean = des_mean, des_std = des_std)
        # self.longterm_exact_predictor.normalize_predictor(aim_mean = train_mean, aim_std = train_std,
        #                                             des_mean = des_mean, des_std = des_std)
        self.longterm_predictor.normalize_predictor(aim_mean = train_mean, aim_std = train_std,
                                                    des_mean = des_mean, des_std = des_std)
        self.onestep_predictor.normalize_predictor(aim_mean = train_mean, aim_std = train_std,
                                                    des_mean = des_mean, des_std = des_std)

    def calculate_max_reward_instance(self, workload):
        """
            供其他函数调用的方法，根据流量值计算出对应的实例数
            利用sample方法
        """
        if workload < 0:
            return 1

        return self._judge_instance_num(workload)

    def __calculate_max_reward_workload(self):
        """
            辅助函数，在对大部分实例（<20）计算
        """
        # 尝试读取result_array
        # 对于过大的实例读取，采用通算方式
        arr_len = 1000
        workload_list = np.linspace(0.1,self.MAX_WKL,arr_len)
        reward_list = []
        res_list = []
        cpu_list = []
        max_reward = -np.inf
        aim_workload = 0
        for per_workload in workload_list:
            cpu_predict_result = self.cpu_predictor.predict_once(per_workload)
            res_predict_result = self.res_predictor.predict_once(per_workload)
            reward = self.get_reward(res_predict_result, cpu_predict_result)
            reward_list.append(reward)
            res_list.append(res_predict_result)
            cpu_list.append(cpu_predict_result)
            if max_reward < reward:
                max_reward = reward
                aim_workload = per_workload
        return aim_workload

    def _judge_instance_num(self, predict_workload):
        """
            输入预测的流量值，输出该流量值下使得奖励函数最大的实例数
        """
        return math.ceil(predict_workload / self.max_workload_count)

    def is_action_basedon_predict_result(self,action=0):
        if self.use_continuous:
            action = np.where(np.max(action) == action)[0][0]
        if action >= 2*self.action_max+1:
            return True
        else:
            return False

    def generate_setting(self,without_ratio=False):
        """根据自身属性产生设置相关内容
        
        """
        if without_ratio:
            return f"{self.workload_name}_{self.train_step}_{self.vali_step}_{self.test_step}_{self.sla}"
        else:
            return f"{self.workload_name}_{self.train_step}_{self.vali_step}_{self.test_step}_{self.sla}_rewardratio_{config_instance.reward_bias_ratio}"

    def get_aim_step(self, cur_mode):
        if cur_mode==TRAIN_MODE:
            return self.train_step
        elif cur_mode == VALI_MODE:
            return self.vali_step
        elif cur_mode == TEST_MODE:
            return self.test_step
        else:
            raise NotImplementedError

    def render(self, mode='human'):
        pass
        
    def close(self):
        pass
    

class APPOEnv(PredictBasicEnv):
    """
        采用了A-SARSA的思想，直接将预测值作为未来结果纳入考量与模拟计算中
        采用最简单的实现方式，直接将响应时间和CPU占用率换成未来的预测结果
            动作空间改会阈值调度
        
        状态空间为：stack的流量向量、CPU占用率、响应时间、stacking（预测器精度SMAPE）、（time of feature）
            流量使用本地正则化方法
            精度应该没办法用本地正则化方法，可以考虑除以100
        
        动作空间为：-v/0/+v，是否直接采用预测结果
        存在最小/最大状态设定
        奖励函数由外源指定
    """
    metadata = {'render.modes': ['human']}
    env_name = 'predict_env_appo' # 该环境为基础云计算模型，状态空间为最简单的版本

    def __init__(self, workload_name = 'Norway',
                     start_point = 30, train_step = 10000, test_step = 3000, vali_step = 3000,
                     des_mean = 10000, des_std = 3000,
                     action_max = 2, stack_point_num = 24, sla=8.,
                     use_burst=True, is_burst_bool=False):
        """
            从指定的流量文件中读取数据，注册到环境中。
            注册奖励函数、状态空间和动作空间
        Args:
            workload_name: 读取的流量的名称
            train_step: 训练段的长度，需要足够的长来保证轨迹的多样性，但是不能够过长
            des_mean, des_std：训练部分流量经过放缩之后对应的均值和方差
            action_max: 最大的调度实例数量v，动作空间为-v~v
            stack_point_num: 状态空间中嵌套的流量点数，状态空间为 （流量向量，预测精度）*stack_point_num+时间特征(4)+CPU占用+响应时间
        """
        super(APPOEnv, self).__init__(workload_name=workload_name, start_point = start_point, 
            train_step=train_step, test_step=test_step, vali_step = vali_step,
            des_mean=des_mean, des_std=des_std, action_max=action_max, stack_point_num=stack_point_num, sla=sla,
            use_burst=use_burst, is_burst_bool=is_burst_bool)
        self.workload_name = workload_name
        # 注册空间，从目前的感觉
        # self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(stack_point_num+2,), dtype=np.float32) 
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(stack_point_num+8,), dtype=np.float32) 
        self.action_space = spaces.Discrete(2*action_max+1)

        # 其他状态
        self.instance = 1
        self.cur_mode = TRAIN_MODE # 0为训练模式，1为测试模式
        # self.max_workload_count = self.__calculate_max_reward_workload()
        # * error_ratio 考虑中的随机值，此状态出现会直接改变响应时间（极大提升），可以用作噪声添加与干扰

    def step(self, action):
        """
            输入动作并执行。
            action 为 2 * action_max + 2，其中最后一个是目标值
            返回state, reward, done, info
        """
        illegal_reward_penalty = 0
        self.instance += action - self.action_max
        if self.instance <= 0:
            self.instance = 1
            illegal_reward_penalty = -3
        elif self.instance > MAX_INS:
            self.instance = MAX_INS
            illegal_reward_penalty = -3

        done = False
        self.current_point += 1
        if self.cur_mode == TRAIN_MODE and self.current_point >= self.train_step - 1:
            done = True
        elif self.cur_mode == VALI_MODE and self.current_point >= self.vali_step - 1:
            done = True
        elif self.cur_mode == TEST_MODE and self.current_point >= self.test_step - 1:
            done = True

        new_state = self._get_state_by_point(self.current_point, self.instance)
        # NOTE ASARSA方法出现严重的泄露问题，奖励值是根据预测结果分配的
        # reward = self.get_reward(self.get_response_time_from_state(new_state),self.get_cpu_utilization_from_state(new_state))

        reward = self.get_reward(self.get_cur_res(),self.get_cur_cpu())

        if self.use_burst: # 根据现有数据判断新状态是否为burst。如果是，则转为激进方法；否则，继续执行该方法
            result = self.burst_detector.is_burst(self.current_point,mode=self.cur_mode)
            if result == 1:
                info = {"is_burst":True}
                done = True # 当前一定是non-burst的。如果由non_burst转为了burst，则
            else:
                info = {"is_burst":False}
        else:
            info = {}
        info.update({"step":self.current_point})
        state = new_state
        reward += illegal_reward_penalty
        return state, reward, done, info # 返回新的状态，从旧状态到新状态的奖励，是否完成，和额外的信息字典

    def _get_state_by_point(self, aim_point, instance):
        """
            给定目标时间点和当时的实例数，返回一个状态
        """
        workload = self._get_noscale_workload_by_point(aim_point)
        # workload = self._get_workload_by_point(aim_point)
        predict_workload = self._get_latest_onestep_prediction_result(aim_point)
        current_features = self._get_features(aim_point)
        per_workload = np.array([predict_workload / instance]).reshape(-1,1)
        if per_workload.item() < self.MAX_WKL:
            cpu_predict_result = self.cpu_predictor.predict_once(per_workload, instance=self.instance)
            res_predict_result = self.res_predictor.predict_once(per_workload, instance=self.instance)
        else:
            cpu_predict_result = self.MAX_CPU
            res_predict_result = self.MAX_RES

        # 进行正则化
        predict_res = self.res_predictor.predict_once(np.array([predict_workload / instance]).reshape(-1,1), instance=self.instance)
        result = np.array([
                    self.get_cpu_state(cpu_predict_result), 
                    self.get_res_state(res_predict_result),
                    self.get_ins_state(self.instance),
                    self.get_res_warning_state(predict_res)])
        result = np.concatenate([result, 
                    workload,
                    current_features,
                    # self.get_onestep_acc_state(current_accuracy),
                    # self.get_longterm_acc_state(longterm_accuracy),
                    ])
        # 计算正式的当前状态
        aim_point = self.current_point
        instance = self.instance
        cur_workload = self.get_latest_workload_by_point(aim_point)
        per_workload = np.array([cur_workload / instance]).reshape(-1,1)
        if per_workload.item() < self.MAX_WKL:
            cpu_predict_result = self.cpu_predictor.predict_once(per_workload, instance=self.instance,use_sampling=True)
            res_predict_result = self.res_predictor.predict_once(per_workload, instance=self.instance,use_sampling=True)
        else:
            cpu_predict_result = self.MAX_CPU
            res_predict_result = self.MAX_RES
        self.cpu_cache[int(per_workload.item())] = cpu_predict_result
        self.res_cache[int(per_workload.item())] = res_predict_result
        return result

    def get_reward(self,res,cpu):
        return custom_reward_penalty(res, cpu,response_sla=self.sla)

    def is_action_basedon_predict_result(self,action=0):
        return False

    def render(self, mode='human'):
        pass
        
    def close(self):
        pass
    
    