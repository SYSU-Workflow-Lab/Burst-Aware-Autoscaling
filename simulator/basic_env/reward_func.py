import numpy as np
from util.constdef import config_instance
# SLA暂定为10ms
def origin_reward_for_basic_env(response_time, cpu, sla=8.):
    """
        A-SARSA版本的reward，代表的是代价而非奖励
    """
    p = 1
    max_reward = (1-np.exp(-p)) / 1e-3
    if cpu >= 1.0:
        reward = max_reward
    else:
        if response_time < sla:
            reward =  (1-np.exp(-p)) / (1-cpu)
        else:
            res_penalty = 1 - response_time / sla
            reward =  (1-np.exp(-p) * res_penalty) / (1-cpu)

        if reward > max_reward:
            reward = max_reward
    return -1 * reward

def custom_reward(response_time, cpu_utilization, response_sla=8, alpha=0.5):
    if cpu_utilization > 100:
        cpu_utilization = 100
    response_value = 1
    if response_time > response_sla:
        response_value = response_sla / response_time
    return alpha * response_value + (1-alpha) * cpu_utilization / 100

def custom_reward_penalty(response_time, cpu_utilization, cpu_threshold = 90, response_sla=8, response_penalty=1,penalty_ratio=8):
    """
    response_sla是联系比较紧密的变量，需要由外部提供
        给定的CPU和res为原来的值，没有经过放缩处理
        penalty_ratio，放缩惩罚值的系数，来调节分配与SLA违约之间的关系
    """
    alpha = config_instance.reward_bias_ratio
    if cpu_utilization > 100:
        cpu_utilization = 100
    if response_time <= 0:
        response_time = 1e-3
    if cpu_utilization <= cpu_threshold: # 奖励线性增加
        cpu_value = cpu_utilization / cpu_threshold
    else: # 奖励指数下降
        cpu_acc = (cpu_utilization-cpu_threshold)/100/(1-cpu_threshold/100 + 1e-6)
        cpu_value = 2 - 1 / (-cpu_acc + 1)
        if cpu_value < 0:
            cpu_value *= penalty_ratio
            if cpu_value < -penalty_ratio: # 做截断
                cpu_value = -penalty_ratio
    response_value = 1
    if response_time > response_sla:
        response_value = (2 - response_sla / response_time) * response_penalty * -1
        response_value *= penalty_ratio
    return alpha * response_value + (1-alpha) * cpu_value

