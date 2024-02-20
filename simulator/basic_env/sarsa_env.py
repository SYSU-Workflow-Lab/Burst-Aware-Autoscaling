from basic_env.predict_env import PredictBasicEnv
import numpy as np
from gym import spaces
from util.constdef import IF_RETRAIN, MAX_WORKLOAD_FACTOR, create_folder_if_not_exist, read_workload_data, get_cpu_predictor, get_res_predictor, read_longterm_prediction_result, parse_workload_data,\
        CACHE_DATA_DIR, TRAIN_MODE, TEST_MODE, VALI_MODE, MAX_RES, MAX_CPU, MAX_WKL, MAX_INS, LONGTERM_EXACT_DIR
from basic_env.reward_func import custom_reward_penalty, origin_reward_for_basic_env
from predictor.performance_predictor import get_origin_data
# SARSA方法需要什么样的环境

class ASARSA_Env(PredictBasicEnv):
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
    env_name = 'baseline_asarsa_sarsa_version' # 该环境为基础云计算模型，状态空间为最简单的版本

    def __init__(self, workload_name = 'Norway',
                     start_point = 30, train_step = 10000, test_step = 3000, vali_step = 3000,
                     des_mean = 10000, des_std = 3000,
                     action_max = 2, stack_point_num = 24, sla=8.,
                     use_burst=True, is_burst=False):
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
        super(ASARSA_Env, self).__init__(workload_name=workload_name, start_point = start_point, 
            train_step=train_step, test_step=test_step, vali_step = vali_step,
            des_mean=des_mean, des_std=des_std, action_max=action_max, stack_point_num=stack_point_num, sla=sla,
            use_burst=use_burst, is_burst_bool=is_burst)
        self.workload_name = workload_name
        # 注册空间，从目前的感觉
        # self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(stack_point_num+2,), dtype=np.float32) 
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32) 
        self.action_space = spaces.Discrete(2*action_max+1)
        # 
        self.cpu_state_num = 10
        self.res_state_num = 6
        # map系列，数组中记录的是每个状态所对应的最大值，将最大值之前的数据映射到同一个状态内
        self.cpu_map = np.arange(10,101,10)
        self.res_map = np.zeros(6) 
        # 根据数据确定响应时间划分
        origin_data = get_origin_data()
        res_value = origin_data['res'].values
        res_value.sort()
        prev_part = res_value[res_value<sla] # 4个部分
        over_part = res_value[res_value>=sla] # 2个部分
        self.res_map[-1] = over_part[-1]
        self.res_map[-2] = over_part[int(len(over_part)/2)]
        self.res_map[-3] = sla
        self.res_map[0] = prev_part[int(len(prev_part)/4*1)]
        self.res_map[1] = prev_part[int(len(prev_part)/4*2)]
        self.res_map[2] = prev_part[int(len(prev_part)/4*3)]

        # 其他状态
        self.instance = 1
        self.cur_mode = TRAIN_MODE # 0为训练模式，1为测试模式
        # self.max_workload_count = self.__calculate_max_reward_workload()
        # * error_ratio 考虑中的随机值，此状态出现会直接改变响应时间（极大提升），可以用作噪声添加与干扰

    def get_discrete_value(self,map_array,aim_value):
        index = 0
        while index < len(map_array) and aim_value > map_array[index]:
            index += 1
        if index >= len(map_array):
            index = len(map_array)-1
        return index

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
            状态需要进行离散化处理
        """
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

        # 获取下一步的CPU预测值和响应时间预测值
        predict_workload = self._get_latest_onestep_prediction_result(aim_point)
        per_workload = np.array([predict_workload / instance]).reshape(-1,1)
        if per_workload.item() < self.MAX_WKL:
            cpu_predict = self.cpu_predictor.predict_once(per_workload, instance=self.instance)
            res_predict = self.res_predictor.predict_once(per_workload, instance=self.instance)
        else:
            cpu_predict = self.MAX_CPU
            res_predict = self.MAX_RES

        # DEBUG 使用简单的算法尝试进行对比的时候 不要 将下面的注释掉
        # cpu_predict = cpu_predict_result
        # res_predict = res_predict_result
        # result = np.array([
        #             self.get_cpu_state(cpu_predict_result), 
        #             self.get_res_state(res_predict_result)])
        result = np.array([self.get_discrete_value(self.cpu_map,cpu_predict),
            self.get_discrete_value(self.res_map,res_predict)])
        # 计算正式的当前状态

        return result

    def get_reward(self,res,cpu):
        # return origin_reward_for_basic_env(res, cpu,sla=self.sla)
        return custom_reward_penalty(res,cpu,response_sla=self.sla)

    def is_action_basedon_predict_result(self,action=0):
        return False

    def render(self, mode='human'):
        pass
        
    def close(self):
        pass
    
# 绘制奖励函数
# reward_list = []
# res_list = []
# cpu_list = []
# for per_workload in workload_list:
#     cpu_predict_result = self.cpu_predictor.predict_once(per_workload, instance=instance)
#     res_predict_result = self.res_predictor.predict_once(per_workload, instance=instance)
#     reward = origin_reward_for_basic_env(res_predict_result,cpu_predict_result/100,sla=35)
#     reward_list.append(reward)
#     res_list.append(res_predict_result)
#     cpu_list.append(cpu_predict_result)
#     if max_reward < reward:
#         max_reward = reward
#         aim_workload = per_workload