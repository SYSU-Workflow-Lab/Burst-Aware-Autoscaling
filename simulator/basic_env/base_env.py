# 最基本的环境，规定了需要实现的方法
import gym
from util.constdef import TRAIN_MODE, VALI_MODE, TEST_MODE
class BaseEnv(gym.Env):
    # 基本五要素
    def __init__(self):
        self.train_step = 0
        self.vali_step = 0
        self.test_step = 0
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def reset(self,cur_mode=TRAIN_MODE):
        raise NotImplementedError

    def render(self, mode='human'):
        pass
        
    def close(self):
        pass

    def _normalized_data(self, des_mean, des_std):
        """
            必须要有一个正则化方法来对流量进行放缩，不然原始流量的波动性不可能够大
        """
        raise NotImplementedError
    
    def is_action_basedon_predict_result(self,action):
        """
            有一个方法来判断某个动作是否依赖于预测结果，以辅助Evaluator产生对应的数据
        """
        raise NotImplementedError
    
    def get_aim_step(self, cur_mode):
        if cur_mode==TRAIN_MODE:
            return self.train_step
        elif cur_mode == VALI_MODE:
            return self.vali_step
        elif cur_mode == TEST_MODE:
            return self.test_step
        else:
            raise NotImplementedError