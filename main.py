import argparse
import logging
from argument_manager import ArgumentManager
from workload_manager import WorkloadManager
from pod_mangaer import PodManager
from simulator.main import logger_register
import numpy as np
import pandas as pd
import datetime
import time
import random
# 线下模拟实验的核心组件
# prerequisite: 
# 1. 需要提供server_list
# 2. server_list中的所有数据必须运行对应的k6终端

def wait_for_specific_time(start_time, interval_time):
    """
    等待指定时间, 单位为s
    """
    end_time = start_time + datetime.timedelta(seconds=interval_time)
    cur_time = datetime.datetime.today()
    while cur_time < end_time:
        time.sleep(1)
        cur_time = datetime.datetime.today()

def run(argument_manager, workload_manager, pod_manager):
    # 包括初始化流量和实例数等，暂未实现
    argument_manager.reset()
    pod_manager.k8s_scale_instance_to_number(argument_manager.config['init_ins_num'])
    prev_workload = 100
    workload_manager.change_workload_number(prev_workload)
    time.sleep(30)
    for i in range(argument_manager.interval_length):
        start_time = datetime.datetime.today()
        logging.info(f"start running episode: {i}/{argument_manager.interval_length} for workload {argument_manager.workload_name}")
        # 逻辑2：流量发送器发生相应改变
        current_workload_num = workload_manager.get_update_workload()
        # 逻辑3：模型管理器向前一步
        ins_num = pod_manager.get_update_pod()
        # 实施这种变动
        if current_workload_num > prev_workload:
            # 流量上升时，先调整数实例数，在调整流量
            cur_ins = pod_manager.k8s_scale_instance_to_number(ins_num)
            logging.info(f"change instance number to {ins_num}, current ins {cur_ins}")
            time.sleep(30)
            act_workload_num = workload_manager.change_workload_number(current_workload_num)
            logging.info(f"change workload count to {act_workload_num} / {current_workload_num}")
        else:
            act_workload_num = workload_manager.change_workload_number(current_workload_num)
            logging.info(f"change workload count to {act_workload_num} / {current_workload_num}")
            time.sleep(30)
            cur_ins = pod_manager.k8s_scale_instance_to_number(ins_num)
            logging.info(f"change instance number to {ins_num}, current ins {cur_ins}")
        prev_workload = current_workload_num
        # 逻辑1：全局时钟向前一步
        argument_manager.step()
        wait_for_specific_time(start_time, argument_manager.config['time_interval'])
    logging.info("finish running")
    _ = pod_manager.get_update_pod()

def main(args):
    argument_manager = ArgumentManager(configure_filename=args.config_filename,
                                        record_filename=args.output_filename)
    # 定义流量产生器
    # 功能：
    # 1. 根据统一时钟读取流量
    # 2. 修改对应的服务器的值
    workload_manager = WorkloadManager(argument_manager)
    # 定义模型产生器
    # 功能：
    # 1. 根据统一时钟，读取当前prometheus的行为
    # 2. 执行动作，修改pod的值
    pod_manager = PodManager(argument_manager)

    # 开始具体执行
    run(argument_manager, workload_manager, pod_manager)

def collect_data(args):
    """
        在实际环境中随机探索，收集数据
    """
    argument_manager = ArgumentManager(configure_filename=args.config_filename,
                                        record_filename=args.output_filename)
    workload_manager = WorkloadManager(argument_manager)
    pod_manager = PodManager(argument_manager)
    argument_manager.reset()
    pass_ins = 10
    pass_workload = 10
    max_workload = 2500

    explore_ins_list = [20,40,100]
    explore_wkl_list = [30,60,100]

    cpu_list = list()
    ins_list = list()
    wkl_list = list()
    err_list = list()
    res_list = list()
    P95_list = list()
    P99_list = list()
    origin_time = datetime.datetime.today()

    for i in range(argument_manager.config['collect_interval_length']):
        start_time = datetime.datetime.today()
        logging.info(f"start running episode: {i}/{argument_manager.config['collect_interval_length']}")
        # 收集数据并记录
        cpu = pod_manager.prometheus_get_cpu()
        ins = pod_manager.prometheus_get_ins()
        wkl = pod_manager.prometheus_get_wkl()
        err = pod_manager.prometheus_get_err()
        res = pod_manager.prometheus_get_res()
        P95_data = pod_manager.prometheus_get_P95()
        P99_data = pod_manager.prometheus_get_P99()
        logging.debug(f"episode {i} data: ins:{ins}, workload:{wkl}, cpu/res:{cpu}/{res}, P95/P99:{P95_data}/{P99_data} err:{err}")
        cpu_list.append(cpu)
        ins_list.append(ins)
        wkl_list.append(wkl)
        err_list.append(err)
        res_list.append(res)
        P95_list.append(P95_data)
        P99_list.append(P99_data)
        data = pd.DataFrame({"cpu":cpu_list, 
                            "res":res_list,
                            "err":err_list,
                            "wkl":wkl_list,
                            "P95":P95_list,
                            "P99":P99_list,
                            "ins":ins_list})
        data.to_csv(f'{origin_time.strftime("%d_%H_%M")}.csv', index=False,mode='w')
        pass_wkl = wkl

        # 产生随机数
        max_perworkload = 100
        min_perworkload = 50
        if random.random() < 0.25:
            min_perworkload = 1
            max_perworkload = 50
        aim_perworkload = random.uniform(min_perworkload,max_perworkload)
        
        max_ins = int(min(100,max_workload/aim_perworkload))
        min_ins = 1
        aim_ins = random.randint(min_ins, max_ins)
        aim_workload = int(aim_perworkload * aim_ins)

        next_ins = aim_ins
        next_wkl = aim_workload
        logging.debug(f"ins:{ins}=>{next_ins}, wkl:{wkl}=>{next_wkl}, explore{next_wkl/next_ins}")
        # 进行具体的调度
        if next_wkl > pass_wkl: # 流量增大时，先扩实例，再调整流量
            pod_manager.k8s_scale_instance_to_number(next_ins)
            time.sleep(20)
            workload_manager.change_workload_number(next_wkl)
        else:# 流量减小时，先减小流量，再调整实例
            workload_manager.change_workload_number(next_wkl)
            time.sleep(20)
            pod_manager.k8s_scale_instance_to_number(next_ins)
        # 等待结果

        wait_for_specific_time(start_time, argument_manager.config['time_interval'])
    logging.info("finish collect_data")

if __name__ == "__main__":
    logger_register(outputFilename="main_simulator.log")
    parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')
    parser.add_argument('--config_filename', type=str, default='data.json', help='The filename of configuration.') 
    parser.add_argument('--output_filename', type=str, default='record.csv', help='The filename of output.') 
    args = parser.parse_args()
    origin_time = datetime.datetime.today()
    args.output_filename = f'{origin_time.strftime("%d_%H_%M")}.csv'
    # 执行具体的调度
    main(args)
    # collect_data(args)
