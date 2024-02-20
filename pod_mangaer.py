# 实现资源管理器
from k6.script.k8sop import K8sOp
from model import ScalerModel
import logging
import requests
import datetime
import traceback
import time
import numpy as np

# prometheus函数相关，通过prometheus获取对应的实例
prometheus_query_api = "http://{prometheus_addr}/api/v1/query?query={query}&time={cur_time}"
# 流量
api_workload = 'sum(rate(istio_requests_total{{destination_workload_namespace="{namespace}",destination_workload="{instance_name}"}}[{time_interval}]))'
# 实例数
api_instance_num = 'count(sum(rate(container_cpu_usage_seconds_total{{image!="",namespace="{namespace}"}}[{time_interval}])) by (pod, namespace))'
# 平均响应时间（from istio）
api_average_response_time_istio = 'sum(rate(istio_request_duration_milliseconds_sum{{destination_workload_namespace="{namespace}",destination_workload="wtyfft-deploy"}}[{time_interval}])) / sum(rate(istio_request_duration_milliseconds_count{{destination_workload_namespace="{namespace}",destination_workload="wtyfft-deploy"}}[{time_interval}]))'
# 平均CPU占用率
api_cpu_utilization_ratio = 'sum(rate(container_cpu_usage_seconds_total{{namespace="{namespace}",container="wtyfft-container-name"}}[{time_interval}])) by (pod_name, namespace) / (sum(container_spec_cpu_quota{{namespace="{namespace}",container="wtyfft-container-name"}} / 100000) by (pod_name, namespace)) * 100'
# 平均错误率
api_error_request_ratio = '100 - (sum(rate(istio_requests_total{{destination_workload_namespace="{namespace}",destination_workload="{instance_name}",response_code="200"}}[{time_interval}])) / sum(rate(istio_requests_total{{destination_workload_namespace="{namespace}",destination_workload="{instance_name}"}}[{time_interval}])))*100'
# P系列
api_tail_latency_istio = 'histogram_quantile({percentile}, sum by (le) (rate(istio_request_duration_milliseconds_bucket{{destination_workload="{instance_name}", destination_workload_namespace="{namespace}"}}[{time_interval}])))'

def generate_query_expression(query_exp, percentile='0.5', namespace='wty-istio', instance_name='wtyfft-deploy', time_interval='1m'):
    return query_exp.format(namespace=namespace, instance_name=instance_name, time_interval=time_interval, percentile=percentile)

def get_prometheus_data(prometheus_addr, query_exp):
    start = datetime.datetime.today()
    start_timestamp = int(time.mktime(start.timetuple()))
    prometheus_query_url = prometheus_query_api.format(prometheus_addr = prometheus_addr,
                                                        query = query_exp,
                                                        cur_time = start_timestamp)
    try:
        request_result = requests.get(prometheus_query_url).json()
    except requests.exceptions.Timeout as e:
        logging.info(f"Exception: get timeout {traceback.format_exc()}")
        return 0

    try:
        result = request_result["data"]['result'][0]['value'][-1]
    except Exception:
        result = 0

    if result == 'NaN':
        return 0
    else:
        return float(result)

class PodManager:
    def __init__(self, argument_manager):
        self.argument_manager = argument_manager
        self.service_name = self.argument_manager.config['service_name']
        self.service_namespace = self.argument_manager.config['service_namespace']
        self.prometheus_addr = self.argument_manager.config['prometheus_addr']
        self.prometheus_query_interval = self.argument_manager.config['prometheus_query_interval'] # 30s/1m
        # kubernetes operator，负责具体的操作等，需要进一步测试
        self.operator = K8sOp()
        # 从argument_manager中读取信息
        # 1. 需要获取prometheus相关的内容
        # 获取prometheus的地址（这部分可以不用动态配置，而是采用超参数的方式进行设置）

        # 2. 需要注册对应的强化学习/其他的模型
        #    这部分模型还需要自行读取对应的已有模型
        #    模型需要完成的功能：根据输入环境，输出得到对应的动作
        self.model = ScalerModel(self.argument_manager)
        self.k8s_scale_instance_to_number(self.argument_manager.config["init_ins_num"])

    def get_update_pod(self):
        # 获取最新的current_point，获取最新的环境变量
        cpu = self.prometheus_get_cpu()
        res = self.prometheus_get_res()
        ins_1 = self.prometheus_get_ins()
        # ins_2 = self.k8s_get_current_ins()
        wkl = self.prometheus_get_wkl()
        err = self.prometheus_get_err()
        # 2. 记录prometheus处得到的数据，并传入到argument_manger中
        self.argument_manager.record({"cpu":cpu, "res":res, "ins":ins_1, "err":err,"wkl":wkl})
        # 3. 将更新后的数据与从AM处获得的状态信息进行组装，传入到模型中
        # 4. 模型中将动作传出
        state = self.argument_manager.get_cur_state(cpu, res, ins_1)
        ins_num = self.model.decide_action_from_state(state, ins_1, cpu, res)
        return ins_num

    def k8s_scale_instance_to_number(self,ins_num):
        try:
            current_instance = self.operator.scale_deployment_to_instance(self.service_name,self.service_namespace,ins_num)
        except Exception as e:
            logging.info(f"get exception {traceback.format_exc()}")
            current_instance = 1
        return current_instance

    def k8s_get_current_ins(self):
        try:
            current_instance = self.operator.get_deployment_instance_num(self.service_name, self.service_namespace)
        except Exception as e:
            logging.info(f"get exception {traceback.format_exc()}")
            current_instance = 1
        return current_instance

    # ! prometheus系列函数
    # 需要的数据
    # 服务相关：namespace, pod_name
    # 具体细节：当前时间（可能需要），间隔时间(30s/1m)
    # ? nan该如何处理
    def prometheus_get_ins(self):
        query_exp = generate_query_expression(api_instance_num, 
                                            namespace=self.service_namespace, 
                                            instance_name=self.service_name,
                                            time_interval=self.prometheus_query_interval)
        return get_prometheus_data(self.prometheus_addr, query_exp)

    def prometheus_get_cpu(self):
        query_exp = generate_query_expression(api_cpu_utilization_ratio, 
                                            namespace=self.service_namespace, 
                                            instance_name=self.service_name,
                                            time_interval=self.prometheus_query_interval)
        return get_prometheus_data(self.prometheus_addr, query_exp)

    def prometheus_get_res(self):
        query_exp = generate_query_expression(api_average_response_time_istio, 
                                            namespace=self.service_namespace, 
                                            instance_name=self.service_name,
                                            time_interval=self.prometheus_query_interval)
        return get_prometheus_data(self.prometheus_addr, query_exp)

    def prometheus_get_wkl(self):
        query_exp = generate_query_expression(api_workload, 
                                            namespace=self.service_namespace, 
                                            instance_name=self.service_name,
                                            time_interval=self.prometheus_query_interval)
        return get_prometheus_data(self.prometheus_addr, query_exp)

    def prometheus_get_err(self):
        query_exp = generate_query_expression(api_error_request_ratio, 
                                            namespace=self.service_namespace, 
                                            instance_name=self.service_name,
                                            time_interval=self.prometheus_query_interval)
        return get_prometheus_data(self.prometheus_addr, query_exp)
    
    def prometheus_get_P95(self):
        query_exp = generate_query_expression(api_tail_latency_istio, 
                                            percentile='0.95',
                                            namespace=self.service_namespace, 
                                            instance_name=self.service_name,
                                            time_interval=self.prometheus_query_interval)
        return get_prometheus_data(self.prometheus_addr, query_exp)

    def prometheus_get_P99(self):
        query_exp = generate_query_expression(api_tail_latency_istio, 
                                            percentile='0.99',
                                            namespace=self.service_namespace, 
                                            instance_name=self.service_name,
                                            time_interval=self.prometheus_query_interval)
        return get_prometheus_data(self.prometheus_addr, query_exp)