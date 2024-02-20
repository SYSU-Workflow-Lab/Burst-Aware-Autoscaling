# 负责实现一个原则上单例的参数管理器(最好不要复制它)
import numpy as np
import json
import os
from simulator.basic_env.predict_env import APPOEnv, PredictBasicEnv
from simulator.util.constdef import VALI_MODE, TEST_MODE, OUTPUT_DIR
import pandas as pd

DQN_MODEL = "DQN"
PPO_MODEL = "PPO"
BURST_MODEL = "burst"
ONESTEP_MODEL = "onestep"
MULTISTEP_MODEL = "multistep"
THRE_MODEL = "threshold"

def read_config(config_filename='data.json'):
    """
        读取json格式的配置信息，转为字典模式返回
        Args:
            config_filename：配置文件，直接放在同文件夹下
        Returns:
            一个字典文件，包含所有相关的配置信息
    """
    with open(config_filename, encoding='utf-8') as f:
        data = json.load(f)
    return data

class ArgumentManager:
    def __init__(self, configure_filename = 'data.json',
        record_filename = 'record.csv' ):
        
        self.record_filename = record_filename
        # 解析JSON
        self.config = read_config(configure_filename)
        self.if_replay = False # 是否开启重播模式
        if self.config['if_replay'] > 0:
            self.if_replay = True
        # JSON里面需要具有的数据：
        workload_name = self.config['hyperparameter'][0]['workload_name']
        start_point = self.config['hyperparameter'][0]['start_point']
        train_step = self.config['hyperparameter'][0]['train_step']
        vali_step = self.config['hyperparameter'][0]['vali_step']
        test_step = self.config['hyperparameter'][0]['test_step']
        des_mean = self.config['hyperparameter'][0]['des_mean']
        des_std = self.config['hyperparameter'][0]['des_std']
        sla = float(self.config['hyperparameter'][0]['sla'])
        action_max = self.config['hyperparameter'][0]['action_max']
        self.workload_name = workload_name
        if self.config['method'] == "ppo":
            self.scaler_model = PPO_MODEL
        elif self.config['method'] == "dqn":
            self.scaler_model = DQN_MODEL
        elif self.config['method'] == "burst":
            self.scaler_model = BURST_MODEL
        elif self.config['method'] == "threshold":
            self.scaler_model = THRE_MODEL
        elif self.config['method'] == "onestep":
            self.scaler_model = ONESTEP_MODEL
        elif self.config['method'] == "multistep":
            self.scaler_model = MULTISTEP_MODEL
        else:
            raise NotImplementedError
        # 设置选择的数据范围
        # * 流量名、模式、开始节点和流量长度将唯一地确认一段流量信息
        if self.config['cur_mode'] == "vali":
            self.cur_mode = VALI_MODE
        elif self.config['cur_mode'] == "test":
            self.cur_mode = TEST_MODE
        else:
            raise NotImplementedError
        self.interval_start = self.config['interval_start'] # 从区间的第50个点开始
        self.interval_length = self.config['interval_length'] # 区间总长度为720
        # 初始化环境
        if self.scaler_model == DQN_MODEL:
            self.env = APPOEnv(workload_name = workload_name, action_max=action_max,
                train_step=train_step, vali_step=vali_step, test_step=test_step, start_point=start_point,
                use_burst=True,sla=sla, des_mean=des_mean, des_std=des_std)
        else:
            self.env = PredictBasicEnv(workload_name = workload_name, action_max=action_max,
                train_step=train_step, vali_step=vali_step, test_step=test_step, start_point=start_point,
                use_burst=True,sla=sla, des_mean=des_mean, des_std=des_std)
        self.env.reset(cur_mode=self.cur_mode)

        # 全局时钟
        self.current_point = 0
        # 要在获取到需要哪些数据后进行
        self.cpu_list = list()
        self.res_list = list()
        self.wkl_list = list()
        self.ins_list = list()
        self.err_list = list()
        
        self.burst_array = None

        if self.scaler_model == PPO_MODEL:
            remark = "burst-aware_burst"
            setting = remark + self.env.generate_setting(without_ratio=True) + "_mode_" + str(self.cur_mode)
            path_dir = os.path.join('simulator',OUTPUT_DIR,setting)
            self.burst_array = self.env.burst_detector.burst_array
            if self.cur_mode == VALI_MODE:
                self.burst_array = self.burst_array[train_step:train_step+vali_step]
            elif self.cur_mode == TEST_MODE:
                self.burst_array = self.burst_array[train_step+vali_step:train_step+vali_step+test_step]
            else:
                raise NotImplementedError
            self.burst_instance_array = np.load(os.path.join(path_dir, 'instance_array.npy'))
            self.burst_workload_array = np.load(os.path.join(path_dir, 'workload_array.npy'))
        elif self.scaler_model == BURST_MODEL or self.scaler_model == ONESTEP_MODEL or self.scaler_model == MULTISTEP_MODEL:
            self.if_replay = True

        if self.if_replay:
            # validator的270行
            if self.scaler_model == BURST_MODEL:
                setting = f"burst_aware_prediction_result_{workload_name}_{self.cur_mode}"
            elif self.scaler_model == ONESTEP_MODEL:
                setting = f"onestep_prediction_result_{workload_name}_{self.cur_mode}"
            elif self.scaler_model == MULTISTEP_MODEL:
                setting = f"longterm_prediction_result_{workload_name}_{self.cur_mode}"
            else:
                if self.scaler_model == PPO_MODEL:
                    use_burst = True
                else:
                    use_burst = False
                env_name = self.env.env_name
                setting = f"{workload_name}_{self.cur_mode}_{use_burst}_burst_env_{env_name}_{self.env.generate_setting()}"
            output_dir_name = os.path.join('simulator',OUTPUT_DIR, setting)
            self.replay_instance_array = np.load(os.path.join(output_dir_name, 'instance_array.npy'))
            self.replay_workload_array = np.load(os.path.join(output_dir_name, 'workload_array.npy'))

    def reset(self):
        self.current_point = 0
        self.env.current_point = self.interval_start + self.current_point
        
    def step(self):
        # 调整全局时钟
        self.current_point += 1
        # 更新环境中的状态
        self.env.current_point = self.interval_start + self.current_point

    def get_is_burst(self):
        if self.burst_array is None:
            return False
        else:
            if self.burst_array[self.env.current_point]==0:
                return False
            else:
                return True

    def get_burst_ins(self):
        # TODO 验证是否对齐，并判断ins是否有问题
        return self.burst_instance_array[self.env.current_point]

    def get_action(self):
        # 重播动作，输出动作数组
        if not self.if_replay:
            raise Exception
        return self.replay_instance_array[self.interval_start + self.current_point]

    def get_current_point(self):
        return self.current_point
    
    def get_current_workload_num(self):
        # 返回当前current_point对应的流量数
        # 要求：是一个整数
        return self.env.get_latest_workload_by_point()

    def get_latest_longterm_prediction_result(self):
        self.env.current_point = self.interval_start + self.current_point
        return self.env._get_latest_longterm_prediction_result(self.env.current_point)[-1]

    def get_latest_onestep_prediction_result(self):
        self.env.current_point = self.interval_start + self.current_point
        return self.env._get_latest_onestep_prediction_result(self.env.current_point)

    def get_cur_state(self, cpu, res, ins):
        """
        输入CPU的绝对值和响应时间的绝对值，根据模型输入对应的状态
        """
        # 考虑调用env._get_state_by_point
        if self.scaler_model == DQN_MODEL or self.scaler_model == PPO_MODEL:
            state = self.env._get_state_by_point(self.env.current_point, ins)
            state[0] = self.env.get_cpu_state(cpu)
            state[1] = self.env.get_res_state(res)
            return state
        else:
            return None

    def record(self, info_dict):
        cpu = info_dict['cpu']
        res = info_dict['res']
        err = info_dict['err']
        wkl = info_dict['wkl']
        ins = info_dict['ins']

        self.cpu_list.append(cpu)
        self.res_list.append(res)
        self.err_list.append(err)
        self.wkl_list.append(wkl)
        self.ins_list.append(ins)
        data = pd.DataFrame({"cpu":self.cpu_list, 
                            "res":self.res_list,
                            "err":self.err_list,
                            "wkl":self.wkl_list,
                            "ins":self.ins_list})
        data.to_csv(self.record_filename, index=False,mode='w')