import urllib.parse
import logging
import datetime
import time
import requests
import math
import numpy as np
import pandas as pd
from log import logger_register
import pytz

local_host = 'localhost:30090'

# 流量
api_workload = 'sum(rate(istio_requests_total{{destination_workload_namespace="{namespace}",destination_workload="{instance_name}"}}[{time_interval}]))'
# 实例数
api_instance_num = 'count(sum(rate(container_cpu_usage_seconds_total{{image!="",namespace="{namespace}"}}[{time_interval}])) by (pod, namespace))'
# 响应时间（个人计算方式）
api_average_response_time_custom = 'sum(rate(http_request_duration_seconds_sum{{kubernetes_namespace="{namespace}",path=~"/fft.*"}}[{time_interval}]))/sum(rate(http_request_duration_seconds_count{{kubernetes_namespace="{namespace}",path=~"/fft.*"}}[{time_interval}]))*1000'
# 响应时间（istio计算方式）
api_average_response_time_istio = 'sum(rate(istio_request_duration_milliseconds_sum{{destination_workload_namespace="{namespace}",destination_workload="wtyfft-deploy"}}[{time_interval}])) / sum(rate(istio_request_duration_milliseconds_count{{destination_workload_namespace="{namespace}",destination_workload="wtyfft-deploy"}}[{time_interval}]))'
# CPU占用率
api_cpu_utilization_ratio = 'sum(rate(container_cpu_usage_seconds_total{{namespace="{namespace}",container="wtyfft-container-name"}}[{time_interval}])) by (pod_name, namespace) / (sum(container_spec_cpu_quota{{namespace="{namespace}",container="wtyfft-container-name"}} / 100000) by (pod_name, namespace)) * 100'
# CPU使用量
api_cpu_usage = 'sum(rate(container_cpu_usage_seconds_total{{namespace="{namespace}",container="wtyfft-container-name"}}[{time_interval}])) by (pod_name, namespace)'
# tail latency
api_tail_latency_custom = 'histogram_quantile({percentile}, sum by (le) (rate(http_request_duration_seconds_bucket{{kubernetes_namespace="{namespace}",app="wtyfft-app",path=~"/fft.*"}}[{time_interval}]))) * 1000'
api_tail_latency_istio = 'histogram_quantile({percentile}, sum by (le) (rate(istio_request_duration_milliseconds_bucket{{destination_workload="{instance_name}", destination_workload_namespace="{namespace}"}}[{time_interval}])))'
# 内存占用率
api_memory_utilization_ratio = 'sum(rate(container_memory_working_set_bytes{{namespace="{namespace}",pod=~"wtyfft.*",container=~"wtyfft.*"}}[{time_interval}]))/sum(container_spec_memory_limit_bytes{{namespace="{namespace}",pod=~"wtyfft.*",container=~"wtyfft.*"}})*100'
# 请求错误率
api_error_request_ratio = '100 - (sum(rate(istio_requests_total{{destination_workload_namespace="{namespace}",destination_workload="{instance_name}",response_code="200"}}[{time_interval}])) / sum(rate(istio_requests_total{{destination_workload_namespace="{namespace}",destination_workload="{instance_name}"}}[{time_interval}])))*100'

# query is string, PromQL expression
# start,end is timestamp, rfc3339| unix_timestamp
# step is duration format(30s,1m) or float number of seconds(1,2)
prometheus_query_range_url = 'http://' + local_host + '/api/v1/query_range?query={query}&start={start}&end={end}&step={step}'

# 函数：产生query_range的API
def generate_api(api_url, percentile='0.5',namespace='wty-istio', instance_name='wtyfft-deploy', time_interval='1m'):
    return api_url.format(namespace=namespace, instance_name=instance_name, time_interval=time_interval, percentile = percentile)

# 函数：解析RESTful API返回的JSON结果

def get_interval(time_interval):
    if time_interval == '30s':
        return 30
    elif time_interval == '1m':
        return 60
    elif time_interval == '2m':
        return 120
    else:
        raise Exception

# 函数：请求API，返回结果，并进行响应的处理
# 参数：开始时间，持续时间，
MAX_ELAPSE_TIME = 600
def get_result_data(request_origin_url, start, elapse_time = MAX_ELAPSE_TIME, time_interval='30s', drop_last=True, percentile = '0.5'):
    expected_interval = get_interval(time_interval)
    # 无论长短，每十分钟进行一轮
    if elapse_time > MAX_ELAPSE_TIME:
        # 进行拆分，拆成多组，然后将这些组的数据拼装
        group_count = elapse_time/MAX_ELAPSE_TIME
        # 如果不能够整除，则增加一个额外项
        result_list = []
        rest_time = elapse_time - MAX_ELAPSE_TIME * math.floor(group_count)
        cache_time = start
        drop_last = True # 只在最后一个开启
        for i in range(math.floor(group_count)):
            if i == math.floor(group_count)-1 and rest_time == 0:
                drop_last = False
            result_list.append(get_result_data(request_origin_url, 
                                               cache_time, 
                                               elapse_time=MAX_ELAPSE_TIME, 
                                               time_interval=time_interval,
                                               drop_last = drop_last))
            cache_time += datetime.timedelta(seconds=MAX_ELAPSE_TIME)
        # should query rest time
        if rest_time > 0:
            result_list.append(get_result_data(request_origin_url, 
                                               cache_time, 
                                               elapse_time=rest_time, 
                                               time_interval=time_interval,
                                               drop_last=False))
        return np.concatenate(result_list)
    else:
        # 计算最终序列应该有多少个点
        expected_data_num = int(elapse_time/expected_interval)
        if elapse_time % expected_interval == 0:
            expected_data_num += 1

        end = start + datetime.timedelta(seconds=elapse_time)
        request_url = generate_api(request_origin_url, percentile = percentile, time_interval=time_interval)
        api_url = prometheus_query_range_url.format(query = request_url, 
                                                    start = int(time.mktime(start.timetuple())), 
                                                    end = int(time.mktime(end.timetuple())), 
                                                    step = time_interval)
        request_result = requests.get(api_url).json()
        if len(request_result["data"]['result'])==0:
            begin_timestamp = int(time.mktime(start.timetuple()))
            end_timestamp = int(time.mktime(end.timetuple()))
            expected_time_index = np.arange(begin_timestamp, end_timestamp+1, expected_interval)
            if expected_time_index[-1] + expected_interval == end_timestamp:
                expected_time_index = np.append(expected_time_index, end_timestamp)
            true_result = np.empty(len(expected_time_index))
            true_result[:] = np.nan
            logging.info("error: zero result")
            if drop_last:
                return true_result[:-1]
            else:
                return true_result

        result = request_result["data"]['result'][0]['values']
        result = np.array(result)
        # 收集第二项
        cache = result.astype('float64')
        if cache.shape[0] < expected_data_num:
            # 构建结果数据集
            begin_timestamp = int(time.mktime(start.timetuple()))
            end_timestamp = int(time.mktime(end.timetuple()))
            expected_time_index = np.arange(begin_timestamp, end_timestamp+1, expected_interval)
            if expected_time_index[-1] + expected_interval == end_timestamp:
                expected_time_index = np.append(expected_time_index, end_timestamp)
            true_result = np.empty(len(expected_time_index))
            true_result[:] = np.nan

            cache_index = 0
            for index, value in enumerate(expected_time_index):
                if cache_index < len(cache) and value == cache[cache_index, 0]:
                    true_result[index] = cache[cache_index, 1]
                    cache_index += 1
            logging.info(f"start time{start} elapse time{elapse_time} got length patch. From {len(result)} to {len(true_result)}")
        else:
            # 如果长度够的话，直接截取最后一个维度即可
            true_result = cache[:,1]

        logging.info(f"start time{start} elapse time{elapse_time}, got length {true_result.shape}")
        if drop_last:
            return true_result[:-1]
        else:
            return true_result

def download_data_and_store_with_name(api_url, start, lasted_time, file_name, percentile = '0.5'):
    result = get_result_data(api_url, start, elapse_time = lasted_time, percentile = percentile)
    np.savetxt(file_name, result, delimiter=",") # 可能含有nan值

data_store = {}
def download_data_and_aggregate_with_pandas(api_url, start, lasted_time, file_name, percentile = '0.5'):
    result = get_result_data(api_url, start, elapse_time = lasted_time, percentile = percentile)
    origin_name = file_name[:file_name.find('.csv')]
    data_store[origin_name] = result

if __name__ == "__main__":
    logger_register()
    # 确定开始时间
    # 1. 最后的秒数必须为0，不然不确定会发生什么事情
    # 2. 第一个点必须确保所有的位置都能找到
    # start_time = "2021-12-25 19:36:00" # 12-25数据
    # start_time = "2021-12-26 14:20:00"
    # start_time = "2021-12-27 10:02:00"
    start_time = "2021-12-29 16:57:00"
    start = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    timezone = pytz.timezone("Asia/Shanghai")
    start = timezone.localize(start) # time zone change
    start = start.astimezone(pytz.utc)
    # 确定持续时间
    lasted_time = 57600
    # lasted_time = 1200
    logging.info(f"start to search for utc time {start} with {lasted_time} s")

    # 获取数据并写入到目标脚本中
    # 实例数
    download_data_and_aggregate_with_pandas(api_instance_num, start, lasted_time, "instance.csv")
    # 流量
    download_data_and_aggregate_with_pandas(api_workload, start, lasted_time, "workload.csv")
    # 平均响应时间-custom
    download_data_and_aggregate_with_pandas(api_average_response_time_custom, start, lasted_time, "response_avg.csv")
    # 平均响应时间-istio
    download_data_and_aggregate_with_pandas(api_average_response_time_istio, start, lasted_time, "response_avg_istio.csv")
    # P95-istio
    download_data_and_aggregate_with_pandas(api_tail_latency_istio, start, lasted_time, "P95_istio.csv", percentile='0.95')
    # P95-custom
    download_data_and_aggregate_with_pandas(api_tail_latency_custom, start, lasted_time, "P95_custom.csv", percentile='0.95')
    # CPU占用率
    download_data_and_aggregate_with_pandas(api_cpu_utilization_ratio, start, lasted_time, "cpu_utilization.csv")
    # 内存占用率
    download_data_and_aggregate_with_pandas(api_memory_utilization_ratio, start, lasted_time, "memory_utilization.csv")
    # 错误率
    download_data_and_aggregate_with_pandas(api_error_request_ratio, start, lasted_time, "error_ratio.csv")
    df = pd.DataFrame({k:v for k,v in data_store.items()})
    df.to_csv('2021_12_29.csv')