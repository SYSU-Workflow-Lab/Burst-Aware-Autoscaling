# 使用requests向指定地址发送API，实现远程操控k6客户端
import requests
import json
import math
import random
import numpy as np
import logging
from datetime import datetime, timedelta
import time
from k8sop import K8sOp
from log import logger_register

local_addr = "localhost:6565"
status_url = "/v1/status"
metric_url = "/v1/metrics"
oper = K8sOp()

def get_status(addr):    
    url = "http://" + addr + status_url
    response_result = requests.get(url,headers={"Content-Type":"application/json"})
    return response_result.json()

def update_state(addr, update_status):
    url = "http://" + addr + status_url
    response_result = requests.patch(url, data=json.dumps(update_status), headers={"Content-Type":"application/json"})
    return response_result.json()

def list_metric(addr):
    url = "http://" + addr + metric_url
    response_result = requests.get(url,headers={"Content-Type":"application/json"})
    return response_result.json()

# 读取配置文件，加载配置信息
def read_config(config_filename='data.json'):
    """
        读取json格式的配置信息，转为字典模式返回
        Args:
            config_filename：配置文件，直接放在同文件夹下
        Returns:
            一个字典文件，包含所有相关的配置信息
    """
    with open(config_filename,"r") as f:
        data = json.load(f)
    return data

# 读取一个离线的流量文件
def read_workload_file(workload_file='test.csv'):
    data = np.genfromtxt(workload_file, dtype=float, delimiter=',', names=True)
    # data = data['view']
    return data


MAX_INSTANCE = 20
MAX_WORKLOAD_PER_INSTNACE = 26 # 10000的情况下，26能压到最大值
OVERFLOW_RATIO = 1.02
MAX_WORKLOAD = MAX_INSTANCE*MAX_WORKLOAD_PER_INSTNACE*OVERFLOW_RATIO

def adjust_server_to_workload_count(server_list, workload_count):
    """
        调整指定服务器到目标流量
    """
    logging.info(f"adjust_server_to_workload_count {server_list} to {workload_count}")

    # 以一定概率修改实例数
    if random.random()<0.5:
        instance_num = random.randint(1,MAX_INSTANCE)
        current_instance = oper.scale_deployment_to_instance('wtyfft-deploy','wty-istio',instance_num)
        logging.info(f'current instance num is {current_instance}')

    for server in server_list:
        # 临时替代，应该要加上-a参数，或者在内部进行设置
        # server = local_addr # 已经加上-a参数

        status = get_status(server)
        max_value = status['data']['attributes']['vus-max']
        actual_value = status['data']['attributes']['vus']
        if actual_value == workload_count:
            continue
        elif workload_count > max_value:
            logging.info(f"meet max_value{max_value}, update to {workload_count}")
            status['data']['attributes'] = {'vus-max':workload_count}
            status = update_state(server, status)
        status['data']['attributes'] = {'vus':workload_count}
        status = update_state(server, status)
        logging.info(f"update server {server} workload from {actual_value} to {workload_count}, response {status}")
    
def adjust_server_to_workload_random(server_list,prev_workload):
    """
        测试用，用以产生随机流量。
        调整指定服务器到目标流量，并以一定概率调整服务的实例数。
        Args:
            server_list: 部署了K6的服务器
        Returns:
            None
        网络交互:
            通过update_state发送API请求给k6服务器
            通过oper.scale_deployment_to_instance调整实例数

    """
    # 获取当前的流量数

    # 以一定概率修改实例数
    if random.random()<0.5:
        max_inst = MAX_INSTANCE
        # 防止突然减小到一个过小的值，导致崩溃
        while max_inst>=2 and prev_workload / max_inst > OVERFLOW_RATIO * MAX_WORKLOAD_PER_INSTNACE:
            max_inst-= 1
        instance_num = random.randint(2,max_inst)
        current_instance = oper.scale_deployment_to_instance('wtyfft-deploy','wty-istio',instance_num)
        logging.info(f'current instance num is {current_instance}')
    else:
        current_instance = oper.get_deployment_instance_num('wtyfft-deploy','wty-istio')
        logging.info(f'current instance num stays {current_instance}')

    workload_count = int(random.random()*OVERFLOW_RATIO*current_instance*MAX_WORKLOAD_PER_INSTNACE/len(server_list))

    logging.info(f"adjust_server_to_workload_count {server_list} to {workload_count}")
    for server in server_list:
        # 临时替代，应该要加上-a参数，或者在内部进行设置
        # server = local_addr
        status = get_status(server)
        max_value = status['data']['attributes']['vus-max']
        actual_value = status['data']['attributes']['vus']
        if actual_value == workload_count:
            continue
        elif workload_count > max_value:
            logging.info(f"meet max_value{max_value}, update to {workload_count}")
            status['data']['attributes'] = {'vus-max':workload_count}
            status = update_state(server, status)
        status['data']['attributes'] = {'vus':workload_count}
        status = update_state(server, status)
        logging.info(f"update server {server} workload from {actual_value} to {workload_count}, response {status}")
    return workload_count*len(server_list)

def adjust_server_to_workload_smooth(server_list,prev_workload):
    """
        用于产生测试流量，得到足够平稳的测试数据，用以满足一般情况下的测试（所以不会考虑极端情况）。
        与随机情况adjust_server_to_workload_random相比，应该会更加稳定一些。
        原理:
        1. 首先，流量的波动会按照一定比例，从而避免马上造成崩溃。
        2. 实例的变动会根据流量波动的比例来进行，
        3. 
        Args:
            server_list: 部署了K6的服务器
        Returns:
            None
        网络交互:
            通过update_state发送API请求给k6服务器
            通过oper.scale_deployment_to_instance调整实例数

    """
    # prev_workload为之前的流量数
    # 基于这个流量确定之后的流量变动，20%的可能性在-10%~10%中间波动，40%在10%~200%波动，40%在-10%~-70%波动。如果碰到下界和上界则重新吞并
    # * 实现方法：实时计算这个概率
    # * 首先确定三个基础概率：0.2,0.4,0.4。
    # * 然后计算出实际概率：mask(基础概率)，根据实际概率使用累加的方式进行划线
    # * 最终投的时候看是否落在实际概率中。
    basic_up_prob, basic_stay_prob, basic_down_prob = 0.4, 0.2, 0.4
    min_change_workload, max_change_workload = 10, MAX_WORKLOAD
    if prev_workload < min_change_workload:
        basic_down_prob = 0
    elif prev_workload >= max_change_workload:
        basic_up_prob = 0

    # 概率变动决定下一次的流量
    total_prob = basic_up_prob + basic_down_prob + basic_stay_prob
    up_prob = basic_up_prob / total_prob
    stay_prob = (basic_up_prob + basic_stay_prob) / total_prob
    choose_prob_value = random.random()
    if choose_prob_value < up_prob: # 流量上升
        next_workload_ratio = random.random()*1.9+0.1
        next_workload = prev_workload * next_workload_ratio
        if next_workload >= MAX_WORKLOAD:
            next_workload = MAX_WORKLOAD
    elif choose_prob_value < stay_prob: # 流量不变
        next_workload_ratio = random.random()*0.2+0.9
        next_workload = prev_workload * next_workload_ratio
        if next_workload >= MAX_WORKLOAD:
            next_workload = MAX_WORKLOAD
        elif next_workload <= 0:
            next_workload = 0
    else: # 流量下降
        next_workload_ratio = random.random()*0.6+0.3
        next_workload = prev_workload * next_workload_ratio
        if next_workload <= 0:
            next_workload = 0
    next_workload = int(next_workload)
    logging.info(f"workload change {next_workload_ratio} from {prev_workload} to {next_workload}")
    # 基于流量的变动确定实例数的变动
    # 50%不变，50%改变
    max_inst = MAX_INSTANCE
    # 防止突然减小到一个过小的值，导致崩溃
    while max_inst>=2 and next_workload / max_inst < OVERFLOW_RATIO * MAX_WORKLOAD_PER_INSTNACE:
        max_inst-= 1
    # 具体的调度实例
    if random.random()<0.5:
        instance_num = random.randint(max_inst,MAX_INSTANCE)
        current_instance = oper.scale_deployment_to_instance('wtyfft-deploy','wty-istio',instance_num)
        logging.info(f'current instance num is {current_instance}')
    else:
        current_instance = oper.get_deployment_instance_num('wtyfft-deploy','wty-istio')
        if current_instance < max_inst:
            current_instance = oper.scale_deployment_to_instance('wtyfft-deploy','wty-istio',max_inst)
        logging.info(f'current instance num stays {current_instance}')

    # 具体的调整流量数
    workload_count = int(random.random()*OVERFLOW_RATIO*current_instance*MAX_WORKLOAD_PER_INSTNACE/len(server_list))

    logging.info(f"adjust_server_to_workload_count {server_list} to {workload_count}")
    for server in server_list:
        # 临时替代，应该要加上-a参数，或者在内部进行设置
        # server = local_addr
        status = get_status(server)
        max_value = status['data']['attributes']['vus-max']
        actual_value = status['data']['attributes']['vus']
        if actual_value == workload_count:
            continue
        elif workload_count > max_value:
            logging.info(f"meet max_value{max_value}, update to {workload_count}")
            status['data']['attributes'] = {'vus-max':workload_count}
            status = update_state(server, status)
        status['data']['attributes'] = {'vus':workload_count}
        status = update_state(server, status)
        logging.info(f"update server {server} workload from {actual_value} to {workload_count}, response {status}")
    return workload_count*len(server_list)

def adjust2workload(server_list,workload_count):
    """
        调整实例数到指定数量上
    """
    for server in server_list:
        # 临时替代，应该要加上-a参数，或者在内部进行设置
        # server = local_addr
        status = get_status(server)
        max_value = status['data']['attributes']['vus-max']
        actual_value = status['data']['attributes']['vus']
        if actual_value == workload_count:
            continue
        elif workload_count > max_value:
            logging.info(f"meet max_value{max_value}, update to {workload_count}")
            status['data']['attributes'] = {'vus-max':workload_count}
            status = update_state(server, status)
        status['data']['attributes'] = {'vus':workload_count}
        status = update_state(server, status)
        logging.info(f"update server {server} workload from {actual_value} to {workload_count}, response {status}")

if __name__ == "__main__":
    logger_register()
    logging.info("workload sclaer work!")
    # 读取信息
    config = read_config()
    # workload_data = read_workload_file(config['workload_record_file']) # 读取文件
    workload_data = np.random.randint(10,MAX_WORKLOAD, size=(100000)) # 产生随机数
    k6_server_list = config['k6_server_addr'] # 进行调整的k6服务端范围
    time_interval = config['time_interval'] # 每个服务点的
    expected_time = config['expected_time']
    assert type(time_interval) == type(2)
    assert type(expected_time) == type(2)
    # 展开循环
    # 循环次数选择：期望时间与实际产生值中较小的那个，防止超过workload
    loop_count = min(math.floor(expected_time / time_interval), len(workload_data))
    start = datetime.now()
    prev_workload = 100
    for i in range(loop_count):
        logging.info(f"loop {i} start")
        # workload = workload_data[i]
        # 调整所有服务器的流量，并对应调整实例数
        prev_workload = adjust_server_to_workload_smooth(k6_server_list,prev_workload)
        # 沉睡直到等待指定时间
        expected_due_time = start + timedelta(seconds=(i+1)*time_interval)
        rest_time = expected_due_time - datetime.now()
        logging.info(f"loop {i} end, sleep for {rest_time.total_seconds()}s ")
        if rest_time.total_seconds()>0:
            time.sleep(rest_time.total_seconds())
