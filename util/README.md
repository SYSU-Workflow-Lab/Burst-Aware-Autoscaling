# 工具函数类型

## prometheus数据收集器

过去的prometheus数据存在着一个问题，它们是一个点一个点来的。其中一种收集脚本如下：

```python
import datetime
import time
import urllib.parse
import requests

req_api = "sum(rate(istio_requests_total{{destination_workload_namespace='weimch-test',reporter='destination',destination_workload='{}'}}[1m]))"
res_api = "sum(delta(istio_request_duration_seconds_sum{{destination_workload_namespace='weimch-test',reporter='destination',destination_workload='{0}'}}[1m]))/sum(delta(istio_request_duration_seconds_count{{destination_workload_namespace='weimch-test',reporter='destination',destination_workload='{0}'}}[1m])) * 1000"
pod_api = "count(sum(rate(container_cpu_usage_seconds_total{{namespace='weimch-test',container_name='{}'}}[1m])) by (pod_name, namespace))"

def fetch_data(api_str, start_time, latsted_time, filename):
    pout = open(filename, "w")
    start = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    encoded_api = urllib.parse.quote_plus(api_str)
    for i in range(0, latsted_time, 30):
        t = start + datetime.timedelta(0, i)
        unixtime = time.mktime(t.timetuple())
        api_url = "http://139.9.57.167:9090/api/v1/query?time={}&query={}".format(unixtime, encoded_api)
        res = requests.get(api_url).json()["data"]
        if "result" in res and len(res["result"]) > 0 and "value" in res["result"][0]:
            v = res["result"][0]["value"]
            sv = str(v[1])
            if sv == "NaN":
                print("0", file=pout)
            else:
                print(sv, file=pout)
        else:
            print("0", file=pout)
    pout.close()
# 01 csvc
start_time = "2019-12-30 23:19:20"
lasted_time = 600
svc_ls = ["csvc", "csvc1", "csvc2", "csvc3"]
prefix = "bo-small"
for svc in svc_ls:
    fetch_data(res_api.format(svc), start_time, lasted_time, "data/{}_{}_res.log".format(prefix, svc))
    fetch_data(req_api.format(svc), start_time, lasted_time, "data/{}_{}_req.log".format(prefix, svc))
    fetch_data(pod_api.format(svc), start_time, lasted_time, "data/{}_{}_pod.log".format(prefix, svc))

```

而[官方API](https://prometheus.io/docs/prometheus/latest/querying/api/)是提供了query_rangeAPI，这个方法能够更快捷的访问所需要的数据。