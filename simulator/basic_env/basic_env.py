import numpy as np
import pandas as pd

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import os
from joblib import load

from basic_env.reward_func import origin_reward_for_basic_env, custom_reward, custom_reward_penalty
from util.norm import local_scale
from util.time_features import time_features
from util.constdef import read_workload_data, get_cpu_predictor, get_res_predictor, read_longterm_prediction_result, parse_workload_data,\
        TRAIN_MODE, TEST_MODE, VALI_MODE, LONGTERM_START_POINT
from predictor.performance_predictor import CpuPredictor, ResPredictor
from predictor.longterm_predictor import LongtermPredictor
from predictor.onestep_predictor import OneStepPredictor
from basic_env.base_env import BaseEnv
# 最基本的模拟环境系统
class BasicEnv(BaseEnv):
    """
        基础的云调度模拟环境，采用gym-style来模拟k8s水平调度所需要的环境

        状态空间为：单位流量、CPU占用率、响应时间
            保证状态空间输出的为正则化后的结果
                其中单位流量进行正则化的依据为profling的结果
        动作空间为：-1/0/+1
        存在最小/最大状态设定
        奖励函数由外源指定
    """
    metadata = {'render.modes': ['human']}
    env_name = 'basic_cloud_env' # 该环境为基础云计算模型，状态空间为最简单的版本

    '''
        涉及到了比较麻烦的问题。
        之前在设计模拟环境的时候没有考虑将训练和测试分开，训练的时候和测试的时候使用的数据是一样的。这显然是不合理的，因为Agent不可能预知需要进行测试的环境，特别是我们的训练对象是针对预测器进行的。
        如果要将训练部分和测试部分区分开，需要解决比较多的问题：
        1. 数据正则化问题。为了使得效果更加明显，我们将原来的流量数据放缩到了一个更高的水平。
           进行了划分之后，需要更加慎重的考虑数据的正则化问题。正则化的目标均值和方差均来自于训练段的原始数据
           测试部分的数据在进行放缩的时候也需要参考训练部分的数据进行放缩，并且分别保存
        2. 状态产生，训练部分和测试部分的数据现在是不一样了。
           目前的想法是在reset部分指定，不指定默认是train，与原来进行兼容。只有设置为是test的时候才会返回test部分的数据。
           test部分的数据和train部分的数据直接使用不同的变量存储以防止搞串，共用除了前缀之外的变量名。
        3. 结果验证代码
           ppo_agent.py部分的代码需要跟着改变。应该只需要改变evalute.py部分的代码，设置连这部分代码可能都不需要改，因为训练的时候肯定是使用训练段进行验证。
           目前的想法是对Evaluator加一个参数，判断是在进行训练或是进行后续的验证。这个参数会传递给env.reset，然后按照2的做法进行。
        4. 兼容再训练的情况。
           考虑到对在线训练的模拟，即在执行测试段数据到中途的时候模型是可能发生改变的。
           不过好像对于模拟环境而言没有什么问题，至少目前来看是没有
        * 注意：边界条件不要串线了
    '''

    def __init__(self, workload_name = 'Norway', start_point = 0, train_step = 10000, test_step = 6000, des_mean = 10000, des_std = 3000, action_max = 2):
        """
            从指定的流量文件中读取数据，注册到环境中。
            注册奖励函数、状态空间和动作空间
        Args:
            workload_name: 读取的流量的名称
            train_step: 训练段的长度，需要足够的长来保证轨迹的多样性，但是不能够过长
            des_mean, des_std:训练部分流量经过放缩之后对应的均值和方差
        """
        self.workload_name = workload_name
        # 注册空间（有待验证）
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Discrete(2*action_max+1)
        # 注册流量读取器
        #   从指定csv文件中读取流量信息
        origin_workload_data = read_workload_data(workload_name)
        self.longterm_predictor = LongtermPredictor(workload_name)
        self.onestep_predictor = OneStepPredictor(workload_name)
        # origin_data, origin_timestamp, origin, pred, true = read_longterm_prediction_result(workload_name)
        origin_timestamp = self.longterm_predictor.origin_timestamp
        longterm_start_point = LONGTERM_START_POINT # 由长期训练部分的len1+len2决定，如果有修改的话需要跨库进行对应的更正（
        self.start_point = start_point  # start_point 标识开始点
        self.current_point = 0  # current_point 标识当前点
        self.train_step = train_step
        self.test_step = test_step
        self.max_step = train_step
        longterm_cur_start_point = longterm_start_point + self.start_point
        assert longterm_cur_start_point + self.train_step + self.test_step <= len(origin_workload_data), "长度不会超标"
        assert start_point + self.train_step + self.test_step <= len(origin_timestamp), "长度不会超标"
        # 环境信息：此处必须往前一步，不然与预测值会对不上
        self.train_workload_data = origin_workload_data[longterm_cur_start_point-1:longterm_cur_start_point-1+self.train_step] 
        self.test_workload_data = origin_workload_data[longterm_cur_start_point-1+self.train_step:longterm_cur_start_point-1+self.train_step+self.test_step]
        self._normalized_data(des_mean, des_std) # 规定范围后执行正则化
        # 时间戳信息
        self.train_timestamp = origin_timestamp[start_point:start_point+train_step]
        self.test_timestamp = origin_timestamp[start_point+train_step:start_point+train_step+test_step]
        # 预测信息
        self.train_longterm_prediction = self.longterm_predictor.pred[start_point:start_point+train_step]
        self.test_longterm_prediction = self.longterm_predictor.pred[start_point+train_step:start_point+train_step+test_step]
        self.train_onestep_prediction = self.onestep_predictor.pred[start_point:start_point+train_step]
        self.test_onestep_prediction = self.onestep_predictor.pred[start_point+train_step:start_point+train_step+test_step]
        #    对数据进行正则化处理
        # 注册资源预测器（流量->CPU占用，流量->响应时间）
        #   使用两个SVM预测器完成预测
        self.cpu_predictor = CpuPredictor() # get_cpu_predictor()
        self.res_predictor = ResPredictor()
        #   正则化
        self.MAX_WKL = 250 # 最大流量250QPS
        self.MAX_CPU = 100 # 最大CPU占用控制在100%
        self.MAX_RES = 80 # 最大响应时间控制在80ms
        # 其他状态
        self.instance = 1 # instance 虽然不出现在状态中，但确实直接受到影响的状态值
        self.cur_mode = TRAIN_MODE # 0为训练模式，1为测试模式
        self.action_max = action_max # 最大调度实例数
        # * error_ratio 考虑中的随机值，此状态出现会直接改变响应时间（极大提升），可以用作噪声添加与干扰

    def step(self, action):
        """
            输入动作并执行。
            action 为 2 * action_max + 1
            返回state, reward, done, info
        """
        self.instance += action - self.action_max
        if self.instance <= 0:
            self.instance = 1

        done = False
        self.current_point += 1
        if self.cur_mode == TRAIN_MODE and self.current_point >= self.train_step - 1:
            done = True
        elif self.cur_mode == TEST_MODE and self.current_point >= self.test_step - 1:
            done = True

        new_state = self._get_state_by_point(self.current_point, self.instance)
        reward = origin_reward_for_basic_env(new_state)

        info = {"step":self.current_point}
        state = new_state
        return state, reward, done, info # 返回新的状态，从旧状态到新状态的奖励，是否完成，和额外的信息字典

    def reset(self, cur_mode = TRAIN_MODE,restart=False):
        """
            重置开始步长
            Args:
                mode: 决定环境处于何种状态。
        """
        # 重置流量信息，得到当前点
        self.current_point = self.start_point
        self.instance = int(self._get_workload_by_point(self.current_point)/self.MAX_WKL)
        self.cur_mode = cur_mode
        return self._get_state_by_point(self.current_point, self.instance)

    def get_response_time_from_state(self, state):
        return state[2]*self.MAX_RES
    
    def get_cpu_utilization_from_state(self, state):
        return state[1]*self.MAX_CPU

    def get_latest_workload_by_point(self, aim_point=None):
        if aim_point == None:
            aim_point = self.current_point

        if self.cur_mode == TRAIN_MODE:
            return self.train_workload_data[aim_point]
        elif self.cur_mode == TEST_MODE:
            return self.test_workload_data[aim_point]
        else:
            raise NotImplementedError

    def _get_workload_by_point(self, aim_point):
        if self.cur_mode == TRAIN_MODE:
            return self.train_workload_data[aim_point].item()
        elif self.cur_mode == TEST_MODE:
            return self.test_workload_data[aim_point].item()
        else:
            raise NotImplementedError

    def _get_state_by_point(self, aim_point, instance):
        """
            给定目标时间点和当时的实例数，返回一个状态
        """
        workload = self._get_workload_by_point(aim_point)
        per_workload = np.array([workload / instance]).reshape(-1,1)
        if per_workload.item() < self.MAX_WKL:
            cpu_predict_result = self.cpu_predictor.predict_once(per_workload)
            res_predict_result = self.res_predictor.predict_once(per_workload)
        else:
            cpu_predict_result = self.MAX_CPU
            res_predict_result = self.MAX_RES

        # 进行正则化
        result = np.array([per_workload.item()/self.MAX_WKL, cpu_predict_result/self.MAX_CPU, res_predict_result/self.MAX_RES])
        return result

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
        # train_mean, train_std = lof_smooth_data(self.train_workload_data) 
        self.train_workload_data = parse_workload_data(self.train_workload_data, 
                                                        aim_mean = train_mean, aim_std = train_std,
                                                        des_mean = des_mean, des_std = des_std)
        self.test_workload_data = parse_workload_data(self.test_workload_data, 
                                                        aim_mean = train_mean, aim_std = train_std,
                                                        des_mean = des_mean, des_std = des_std)
        self.longterm_predictor.normalize_predictor(aim_mean = train_mean, aim_std = train_std,
                                                    des_mean = des_mean, des_std = des_std)
        self.onestep_predictor.normalize_predictor(aim_mean = train_mean, aim_std = train_std,
                                                    des_mean = des_mean, des_std = des_std)

    def is_action_basedon_predict_result(self,action=0):
        return False

    def render(self, mode='human'):
        pass
        
    def close(self):
        pass
    
class BasicStackEnv(BaseEnv):
    """
        基础的云调度模拟环境，但是稍微有些不同
        区别：与之前模拟环境的区别在于在状态中使用了多个点的流量(h)
            因此，在产生状态的时候需要稍微修改一下，在生成新数据的时候需要多往前读h-1个点以保持一致
        
        状态空间为：stack的流量向量、CPU占用率、响应时间
            流量使用本地正则化方法
        
        动作空间为：-v/0/+v
        存在最小/最大状态设定
        奖励函数由外源指定
    """
    metadata = {'render.modes': ['human']}
    env_name = 'basic_stack_env' # 该环境为基础云计算模型，状态空间为最简单的版本

    def __init__(self, workload_name = 'Norway',
                     start_point = 0, train_step = 10000, test_step = 3000, vali_step = 3000,
                     des_mean = 10000, des_std = 3000,
                     action_max = 2, stack_point_num = 6):
        """
            从指定的流量文件中读取数据，注册到环境中。
            注册奖励函数、状态空间和动作空间
        Args:
            workload_name: 读取的流量的名称
            train_step: 训练段的长度，需要足够的长来保证轨迹的多样性，但是不能够过长
            des_mean, des_std：训练部分流量经过放缩之后对应的均值和方差
            action_max: 最大的调度实例数量v，动作空间为-v~v
            stack_point_num: 状态空间中嵌套的流量点数，状态空间为 流量向量+CPU占用+响应时间
        """
        assert type(action_max) == type(1)
        assert type(stack_point_num) == type(1)

        self.workload_name = workload_name
        # 注册空间，从目前的感觉
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(stack_point_num+2,), dtype=np.float32) 
        self.action_space = spaces.Discrete(2*action_max+1)
        self.stack_point_num = stack_point_num
        self.action_max = action_max
        # 注册流量读取器
        #   从指定csv文件中读取流量信息
        origin_workload_data = read_workload_data(workload_name)
        self.longterm_predictor = LongtermPredictor(workload_name)
        self.onestep_predictor = OneStepPredictor(workload_name)
        # origin_data, origin_timestamp, origin, pred, true = read_longterm_prediction_result(workload_name)
        origin_timestamp = self.longterm_predictor.origin_timestamp
        longterm_start_point = LONGTERM_START_POINT # 由长期训练部分的len1+len2决定，如果有修改的话需要跨库进行对应的更正（
        self.start_point = start_point  # start_point 标识开始点
        self.current_point = 0  # current_point 标识当前点
        self.train_step = train_step
        self.vali_step = vali_step
        self.test_step = test_step
        self.max_step = train_step
        longterm_cur_start_point = longterm_start_point + self.start_point
        assert longterm_cur_start_point + self.train_step + self.vali_step + self.test_step <= len(origin_workload_data), "长度不会超标"
        assert start_point + self.train_step + self.vali_step + self.test_step <= len(origin_timestamp), "长度不会超标"
        # 环境信息：此处必须往前一步，不然与预测值会对不上
        self.train_workload_data = origin_workload_data[longterm_cur_start_point-1-(stack_point_num-1):longterm_cur_start_point-1+self.train_step] 
        self.test_workload_data = origin_workload_data[longterm_cur_start_point-1+self.train_step-(stack_point_num-1):longterm_cur_start_point-1+self.train_step+self.vali_step+self.test_step]
        self._normalized_data(des_mean, des_std) # 规定范围后执行正则化
        # 时间戳信息
        self.train_timestamp = origin_timestamp[start_point:start_point+train_step]
        self.test_timestamp = origin_timestamp[start_point+train_step:start_point+train_step+vali_step+test_step]
        self.train_time_features = time_features(self.train_timestamp)
        self.test_time_features = time_features(self.test_timestamp)
        # 预测信息
        self.train_longterm_prediction = self.longterm_predictor.pred[start_point:start_point+train_step]
        self.test_longterm_prediction = self.longterm_predictor.pred[start_point+train_step:start_point+train_step+vali_step+test_step]
        self.train_onestep_prediction = self.onestep_predictor.pred[start_point:start_point+train_step]
        self.test_onestep_prediction = self.onestep_predictor.pred[start_point+train_step:start_point+train_step+vali_step+test_step]
        #    对数据进行正则化处理
        # 注册资源预测器（流量->CPU占用，流量->响应时间）
        #   使用两个SVM预测器完成预测
        self.cpu_predictor = CpuPredictor() # get_cpu_predictor()
        self.res_predictor = ResPredictor()
        #   正则化
        self.MAX_WKL = 250 # 最大流量250QPS
        self.MAX_CPU = 100 # 最大CPU占用控制在100%
        self.MAX_RES = 80 # 最大响应时间控制在80ms
        # 其他状态
        self.instance = 1 # instance 虽然不出现在状态中，但确实直接受到影响的状态值
        self.cur_mode = TRAIN_MODE # 0为训练模式，1为测试模式
        # * error_ratio 考虑中的随机值，此状态出现会直接改变响应时间（极大提升），可以用作噪声添加与干扰

    def step(self, action):
        """
            输入动作并执行。
            action 为 2 * action_max + 1
            返回state, reward, done, info
        """
        self.instance += action - self.action_max
        if self.instance <= 0:
            self.instance = 1

        done = False
        self.current_point += 1
        if self.cur_mode == TRAIN_MODE and self.current_point >= self.train_step - 1:
            done = True
        elif self.cur_mode == VALI_MODE and self.current_point >= self.vali_step - 1:
            done = True
        elif self.cur_mode == TEST_MODE and self.current_point >= self.test_step - 1:
            done = True

        new_state = self._get_state_by_point(self.current_point, self.instance)
        reward = custom_reward_penalty(self.get_response_time_from_state(new_state),self.get_cpu_utilization_from_state(new_state))

        info = {"step":self.current_point}
        state = new_state
        return state, reward, done, info # 返回新的状态，从旧状态到新状态的奖励，是否完成，和额外的信息字典

    def reset(self, cur_mode = TRAIN_MODE,restart=False):
        """
            重置开始步长
            Args:
                mode: 决定环境处于何种状态。
        """
        # 重置流量信息，得到当前点
        self.current_point = 0
        self.instance = int(self.get_latest_workload_by_point(self.current_point)/self.MAX_WKL)
        self.cur_mode = cur_mode
        return self._get_state_by_point(self.current_point, self.instance)
    
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

    def _get_state_by_point(self, aim_point, instance):
        """
            给定目标时间点和当时的实例数，返回一个状态
        """
        workload = self._get_workload_by_point(aim_point)
        current_workload = self.get_latest_workload_by_point(aim_point)
        per_workload = np.array([current_workload / instance]).reshape(-1,1)
        if per_workload.item() < self.MAX_WKL:
            cpu_predict_result = self.cpu_predictor.predict_once(per_workload)
            res_predict_result = self.res_predictor.predict_once(per_workload)
        else:
            cpu_predict_result = self.MAX_CPU
            res_predict_result = self.MAX_RES

        # 进行正则化
        result = np.array([cpu_predict_result/self.MAX_CPU, res_predict_result/self.MAX_RES])
        result = np.concatenate([result,workload])
        return result
    
    def get_response_time_from_state(self, state):
        return state[1]*self.MAX_RES
    
    def get_cpu_utilization_from_state(self, state):
        return state[0]*self.MAX_CPU

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
        # train_mean, train_std = lof_smooth_data(self.train_workload_data) 
        self.train_workload_data = parse_workload_data(self.train_workload_data, 
                                                        aim_mean = train_mean, aim_std = train_std,
                                                        des_mean = des_mean, des_std = des_std)
        self.test_workload_data = parse_workload_data(self.test_workload_data, 
                                                        aim_mean = train_mean, aim_std = train_std,
                                                        des_mean = des_mean, des_std = des_std)
        self.longterm_predictor.normalize_predictor(aim_mean = train_mean, aim_std = train_std,
                                                    des_mean = des_mean, des_std = des_std)
        self.onestep_predictor.normalize_predictor(aim_mean = train_mean, aim_std = train_std,
                                                    des_mean = des_mean, des_std = des_std)

    def is_action_basedon_predict_result(self,action=0):
        return False

    def render(self, mode='human'):
        pass
        
    def close(self):
        pass
    
    