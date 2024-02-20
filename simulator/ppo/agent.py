"""
This project use source code from eRL_demo_PPOinSingleFile.py 
from [ElegantRL](https://github.com/AI4Finance-Foundation/ElegantRL), 
copyright Yonv1943曾伊言
licensed under the Apache 2.0 license. Followed by the whole Apache 2.0 license text.
"""
"""agent.py"""
import torch

from copy import deepcopy
import numpy as np
from ppo.net import CriticAdv, ActorPPO, ActorDiscretePPO
from typing import Tuple
class AgentPPO:
    def __init__(self):
        super().__init__()
        self.state = None
        self.device = None
        self.action_dim = None
        self.get_obj_critic = None

        self.criterion = torch.nn.SmoothL1Loss()
        self.cri = self.cri_target = self.if_use_cri_target = self.cri_optim = self.ClassCri = None
        self.act = self.act_target = self.if_use_act_target = self.act_optim = self.ClassAct = None

        '''init modify'''
        self.ClassCri = CriticAdv
        self.ClassAct = ActorPPO

        self.ratio_clip = 0.2  # ratio.clamp(1 - clip, 1 + clip)
        self.lambda_entropy = 0.02  # could be 0.01~0.05
        self.lambda_gae_adv = 0.98  # could be 0.95~0.99, GAE (Generalized Advantage Estimation. ICLR.2016.)
        self.get_reward_sum = None  # self.get_reward_sum_gae if if_use_gae else self.get_reward_sum_raw
        self.trajectory_list = None

    def init(self, net_dim, state_dim, action_dim, learning_rate=1e-4, if_use_gae=False, gpu_id=0):
        """
            设置了设备，actor-critic，是否使用GAE，以及目标网络和优化器
        """
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
        self.trajectory_list = list()
        self.get_reward_sum = self.get_reward_sum_gae if if_use_gae else self.get_reward_sum_raw

        self.cri = self.ClassCri(net_dim, state_dim, action_dim).to(self.device)
        self.act = self.ClassAct(net_dim, state_dim, action_dim).to(self.device) if self.ClassAct else self.cri
        self.cri_target = deepcopy(self.cri) if self.if_use_cri_target else self.cri
        self.act_target = deepcopy(self.act) if self.if_use_act_target else self.act

        self.cri_optim = torch.optim.Adam(self.cri.parameters(), learning_rate)
        self.act_optim = torch.optim.Adam(self.act.parameters(), learning_rate) if self.ClassAct else self.cri
        del self.ClassCri, self.ClassAct

    def select_action(self, state):
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
        actions, noises = self.act.get_action(states)
        return actions[0].detach().cpu().numpy(), noises[0].detach().cpu().numpy()

    def explore_env(self, env, target_step):
        trajectory_temp = list()

        state = self.state
        last_done = 0
        # print('explore_env')
        # for i in trange(target_step):
        for i in range(target_step):
            action, noise = self.select_action(state)
            next_state, reward, done, _ = env.step(np.tanh(action))
            trajectory_temp.append((state, reward, done, action, noise))
            if done:
                state = env.reset(restart=False)
                last_done = i
            else:
                state = next_state
        self.state = state

        '''splice list'''
        trajectory_list = self.trajectory_list + trajectory_temp[:last_done + 1]
        self.trajectory_list = trajectory_temp[last_done:]
        return trajectory_list

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
        # 更新网络
        # buf_开头
        # 
        with torch.no_grad():
            buf_len = buffer[0].shape[0] # buffer的长度
            buf_state, buf_action, buf_noise, buf_reward, buf_mask = [ten.to(self.device) for ten in buffer] # 重新对buffer内的数据进行封装
            # (ten_state, ten_action, ten_noise, ten_reward, ten_mask) = buffer

            '''get buf_r_sum, buf_logprob'''
            bs = 2 ** 10  # set a smaller 'BatchSize' when out of GPU memory.
            buf_value = [self.cri_target(buf_state[i:i + bs]) for i in range(0, buf_len, bs)]
            buf_value = torch.cat(buf_value, dim=0)
            buf_logprob = self.act.get_old_logprob(buf_action, buf_noise) # 获得根据旧策略采样得到的动作的log_prob

            buf_r_sum, buf_advantage = self.get_reward_sum(buf_len, buf_reward, buf_mask, buf_value)  # detach()
            buf_advantage = (buf_advantage - buf_advantage.mean()) / (buf_advantage.std() + 1e-5)
            del buf_noise, buffer[:]

        '''PPO: Surrogate objective of Trust Region'''
        # 正式开始执行PPO
        obj_critic = obj_actor = None
        # print('update network')
        # for _ in trange(int(buf_len / batch_size * repeat_times)):
        for _ in range(int(buf_len / batch_size * repeat_times)):
            # 抽样，随机抽取buffer中的一个点
            indices = torch.randint(buf_len, size=(batch_size,), requires_grad=False, device=self.device)

            state = buf_state[indices]
            action = buf_action[indices]
            r_sum = buf_r_sum[indices]
            logprob = buf_logprob[indices]
            advantage = buf_advantage[indices]

            # 使用新的pi，采样新的logprob和entropy
            new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)  # it is obj_actor
            ratio = (new_logprob - logprob.detach()).exp() # 计算ratio
            surrogate1 = advantage * ratio # 原始版本
            surrogate2 = advantage * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip) # clip版本
            obj_surrogate = -torch.min(surrogate1, surrogate2).mean() # 取最小的那个
            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy # 损失函数
            self.optim_update(self.act_optim, obj_actor)

            value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, r_sum) / (r_sum.std() + 1e-6)
            self.optim_update(self.cri_optim, obj_critic)
            # 如果cri_target是deepcopy的，执行soft_update
            self.soft_update(self.cri_target, self.cri, soft_update_tau) if self.cri_target is not self.cri else None

        a_std_log = getattr(self.act, 'a_std_log', torch.zeros(1))
        if obj_critic == None:
            critic_value = 0
        else:
            critic_value = obj_critic.item()
        if obj_actor == None:
            actor_value = 0
        else:
            actor_value = obj_actor.item()
        return critic_value, actor_value, a_std_log.mean().item()  # logging_tuple

    def get_reward_sum_raw(self, buf_len, buf_reward, buf_mask, buf_value) -> Tuple[torch.Tensor, torch.Tensor]:
        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # reward sum

        pre_r_sum = 0
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
        buf_advantage = buf_r_sum - (buf_mask * buf_value[:, 0])
        return buf_r_sum, buf_advantage

    def get_reward_sum_gae(self, buf_len, ten_reward, ten_mask, ten_value) -> Tuple[torch.Tensor, torch.Tensor]:
        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # old policy value
        buf_advantage = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # advantage value

        pre_r_sum = 0
        pre_advantage = 0  # advantage value of previous step
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = ten_reward[i] + ten_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
            buf_advantage[i] = ten_reward[i] + ten_mask[i] * (pre_advantage - ten_value[i])  # fix a bug here
            pre_advantage = ten_value[i] + buf_advantage[i] * self.lambda_gae_adv
        return buf_r_sum, buf_advantage

    @staticmethod
    def optim_update(optimizer, objective):
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

    @staticmethod
    def soft_update(target_net, current_net, tau):
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            # 加权贴近
            tar.data.copy_(cur.data.__mul__(tau) + tar.data.__mul__(1.0 - tau))

from tqdm import trange
class AgentDiscretePPO(AgentPPO):
    def __init__(self):
        super().__init__()
        self.ClassAct = ActorDiscretePPO

    def explore_env(self, env, target_step):
        trajectory_temp = list()

        state = self.state
        last_done = 0
        # print('explore env')
        # for i in trange(target_step):
        reward_list = []
        workload_list = []
        action_list = []
        ins_list = []
        res_list = []
        cpu_list = []
        for i in range(target_step):
            action, a_prob = self.select_action(state)  # different
            a_int = int(action)  # different
            next_state, reward, done, _ = env.step(a_int)  # different

            ins_list.append(env.instance) # 执行完动作后的实例数
            res_list.append(env.get_cur_res())
            cpu_list.append(env.get_cur_cpu())
            reward_list.append(reward)
            workload_list.append(env.get_latest_workload_by_point())
            action_list.append(action)

            trajectory_temp.append((state, reward, done, a_int, a_prob))  # different

            if done:
                state = env.reset(restart=False)
                last_done = i
            else:
                state = next_state
        self.state = state
        # '''splice list'''
        trajectory_list = self.trajectory_list + trajectory_temp[:last_done + 1]
        self.trajectory_list \
            = trajectory_temp[last_done:]
        return trajectory_list
