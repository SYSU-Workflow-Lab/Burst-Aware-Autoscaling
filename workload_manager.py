# 负责实现流量管理部分
from ctypes.wintypes import LONG
import json
import requests
import logging
import traceback

local_addr = "localhost:6565"
status_url = "/v1/status"
metric_url = "/v1/metrics"
LONGER_TIMEOUT = 5

def get_status(addr):
    url = "http://" + addr + status_url
    try:
        response_result = requests.get(url,headers={"Content-Type":"application/json"})
    except requests.exceptions.Timeout as e:
        logging.info(f"Exception: get timeout {traceback.format_exc()}")
    return response_result.json()

def update_state(addr, update_status):
    url = "http://" + addr + status_url
    try:
        response_result = requests.patch(url, data=json.dumps(update_status), headers={"Content-Type":"application/json"})
    except requests.exceptions.Timeout as e:
        logging.info(f"Exception: get timeout {traceback.format_exc()}")
    return response_result.json()

def list_metric(addr):
    url = "http://" + addr + metric_url
    try:
        response_result = requests.get(url,headers={"Content-Type":"application/json"})
    except requests.exceptions.Timeout as e:
        logging.info(f"Exception: get timeout {traceback.format_exc()}")
    return response_result.json()


class WorkloadManager:
    def __init__(self, argument_manager):
        self.argument_manager = argument_manager

        # 读取相关信息
        # 1. 读取k6服务器列表
        self.server_list = self.argument_manager.config['k6_server_addr']
        # 2. 读取对应的流量发送列表
    
    def change_workload_number(self, aim_workload_num):
        """
        修改当前发送的流量数量为指定值
        可以参考k6/script中对应的发送片段
        返回最终实际修改的结果
        """
        server_list = self.server_list
        workload_count = int(aim_workload_num) // len(server_list)
        for server in server_list:
            # 临时替代，应该要加上-a参数，或者在内部进行设置
            # server = local_addr
            status = get_status(server)
            max_value = status['data']['attributes']['vus-max']
            actual_value = status['data']['attributes']['vus']

            if actual_value == workload_count:
                # 如果当前流量满足，则无须继续处理
                continue
            elif workload_count > max_value:
                # 如果当前流量最大值不够，则需要先扩大最大值(PS：绝对不能同时扩展最大值和流量数)
                logging.debug(f"meet max_value{max_value}, update to {workload_count}")
                status['data']['attributes'] = {'vus-max':workload_count}
                status = update_state(server, status)

            status['data']['attributes'] = {'vus':workload_count}
            status = update_state(server, status)
            logging.debug(f"update server {server} workload from {actual_value} to {workload_count}, response {status}")
        return workload_count*len(server_list)

    def get_update_workload(self):
        # 1. 从argument_manager中获取最新的current_point
        # 2. 根据获取到的current_point访问得到流量数（或者从AM拿到这个流量）
        current_workload_num = self.argument_manager.get_current_workload_num()
        return current_workload_num
        