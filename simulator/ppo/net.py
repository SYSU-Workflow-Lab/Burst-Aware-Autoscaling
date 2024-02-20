"""
This project use source code from eRL_demo_PPOinSingleFile.py 
from [ElegantRL](https://github.com/AI4Finance-Foundation/ElegantRL), 
copyright Yonv1943曾伊言
licensed under the Apache 2.0 license. Followed by the whole Apache 2.0 license text.
"""
"""net.py"""
import torch
import torch.nn as nn
import numpy as np
import random
from util.constdef import ACTION_MAX

class ActorPPO(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                 nn.Linear(mid_dim, action_dim), )

        # the logarithm (log) of standard deviation (std) of action, it is a trainable parameter
        self.a_logstd = nn.Parameter(torch.zeros((1, action_dim)) - 0.5, requires_grad=True)
        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))

    def forward(self, state):
        return self.net(state).tanh()  # action.tanh()

    def get_action(self, state):
        a_avg = self.net(state)
        a_std = self.a_logstd.exp()

        noise = torch.randn_like(a_avg)
        action = a_avg + noise * a_std
        return action, noise

    def get_logprob_entropy(self, state, action):
        a_avg = self.net(state)
        a_std = self.a_logstd.exp()

        delta = ((a_avg - action) / a_std).pow(2) * 0.5
        logprob = -(self.a_logstd + self.sqrt_2pi_log + delta).sum(1)  # new_logprob

        dist_entropy = (logprob.exp() * logprob).mean()  # policy entropy
        return logprob, dist_entropy

    def get_old_logprob(self, _action, noise):  # noise = action - a_noise
        delta = noise.pow(2) * 0.5
        return -(self.a_logstd + self.sqrt_2pi_log + delta).sum(1)  # old_logprob


class ActorDiscretePPO(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                 nn.Linear(mid_dim, action_dim))
        self.action_dim = action_dim
        self.soft_max = nn.Softmax(dim=-1)
        self.Categorical = torch.distributions.Categorical
        self.action_max = ACTION_MAX
        self.is_action_mask = False # 当前是否开启动作选择
        self.if_use_multistep = False # 是否开启多步模式

    def forward(self, state):
        action_val = self.net(state)
        if self.is_action_mask:
            action_val[:,2*self.action_max+1:] = 0
        return action_val  # action_prob without softmax

    def get_action(self, state):
        if self.is_action_mask:
            a_val = self.forward(state)
            a_prob_part = self.soft_max(a_val[0,:2*self.action_max+1])
            a_prob = torch.zeros_like(a_val)
            a_prob[0,:2*self.action_max+1] = a_prob_part
        else:
            a_prob = self.soft_max(self.net(state)) # 转成概率表示形式
        # action = Categorical(a_prob).sample()
        # 正常情况
        samples_2d = torch.multinomial(a_prob, num_samples=1, replacement=True) # 抽样
        action = samples_2d.reshape(state.size(0))
        if self.if_use_multistep and not self.is_action_mask:
            # 如果开启多步模式，且没有动作屏障，则此时进入第二阶段的训练
            origin_prob = self.net(state) # 拿到原始的概率
            prev_prob_origin = origin_prob[:,:2*self.action_max+1] # 前段
            prev_prob = self.soft_max(prev_prob_origin)
            after_prob_origin = origin_prob[:,2*self.action_max+1:] # 后段
            after_prob = self.soft_max(after_prob_origin)

            # 弥补模式的退出：如果新的部分最大的动作在概率上接近之前的最大值的30%，则退出弥补模式
            # 判断是否进入弥补模式
            if torch.max(prev_prob) * 0.8 > torch.max(after_prob):
                # 进入弥补模式下，有50%的概率使用之前的数据，有50%的概率使用之后的数据。单独作用softmax
                if random.random() > 0.5:
                    action_prob = prev_prob
                    bias = 0
                else:
                    action_prob = after_prob
                    bias = 2*self.action_max+1
                samples_2d = torch.multinomial(action_prob, num_samples=1, replacement=True) # 抽样
                action = samples_2d.reshape(state.size(0)) + bias
            else:
                self.if_use_multistep = False
                samples_2d = torch.multinomial(a_prob, num_samples=1, replacement=True) # 抽样
                action = samples_2d.reshape(state.size(0))
        return action, a_prob

    def get_logprob_entropy(self, state, a_int):
        a_prob = self.soft_max(self.net(state))
        dist = self.Categorical(a_prob)
        return dist.log_prob(a_int), dist.entropy().mean()

    def get_old_logprob(self, a_int, a_prob):
        dist = self.Categorical(a_prob)
        return dist.log_prob(a_int)


class CriticAdv(nn.Module):
    def __init__(self, mid_dim, state_dim, _action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                 nn.Linear(mid_dim, 1))

    def forward(self, state):
        return self.net(state)  # Advantage value
