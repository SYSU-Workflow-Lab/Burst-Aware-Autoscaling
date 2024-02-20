import os
import pandas as pd
import numpy as np
# 该文件的目标：整合同一个文件中的所有数据
# 1. 列出文件夹下的所有同一个名字的文件
name_list = ["请求错误率", "实例数", "average response time", "CPU 占用率","CPU使用量","Workload"]
# 2. 使用pandas读取这些文件，返回多个pd.DataFrame组成的列表
# 3. 进行处理。drop_duplicate+drop_na

def get_data_from_name(file_name, dir_name):
    """
        输入name_list中的文件，搜索所有包含该名字的文件
    """
    all_file_list = os.listdir(dir_name)
    aim_file_list = [file for file in all_file_list if file.find(file_name)!=-1]
    pd_data_list = []
    for file in aim_file_list:
        pd_data_list.append(pd.read_csv(os.path.join(dir_name,file)))
    total = pd.concat(pd_data_list)
    return total

def get_data_from_dir(dir_name):
    """
    Args:
        dir_name: 目录名称
    Returns:
        pd.DataFrame
        包含instance, workload, cpu_utilization, response_avg_istio, error_ratio, total_cpu_usage
    """
    # 对于每一个名字，获取一个DataFrame，加入到对应的目录中
    file_name2col_name={
        "请求错误率" : "error_ratio",
        "实例数" : "instance",
        "Workload" : "workload",
        "average response time" : "response_avg_istio",
        "CPU 占用率" : "cpu_utilization",
        "CPU使用量" : "total_cpu_usage"
    }
    #枚举name_list中的所有名字
    store = {}
    aim_interval = 10
    for file_name in name_list:
        print(f"{file_name} is merging")
        cache = get_data_from_name(file_name,dir_name)
        # 按时间维排序
        cache['Time'] = pd.to_datetime(cache['Time'])
        cache = cache.sort_values(by='Time')
        # 新建一个result，做Date上的超集，进行填充nan然后写回真值
        start = cache['Time'].iloc[0]
        end = cache['Time'].iloc[-1]
        interval = cache['Time'].iloc[1] - cache['Time'].iloc[0]
        result = pd.DataFrame(index=pd.date_range(start=start,end=end,freq=f'{interval.seconds}s'),columns=[cache.columns[1]])
        result_pivot = 0
        pivot = 0
        while pivot < len(cache):
            # 写入result数据
            while cache['Time'].iloc[pivot] != result.index[result_pivot]:
                result_pivot += 1
            result.iloc[result_pivot] = cache.iloc[pivot][1]
            # 移动到下一个可靠的数据
            pivot += 1
            while pivot < len(cache) and cache['Time'].iloc[pivot-1] == cache['Time'].iloc[pivot]:
                pivot += 1

        aim_name = file_name2col_name[file_name]
        if interval == 5:
            index_filter = np.arange(0,len(result),2)
            result = result.iloc[index_filter]
        store[aim_name] = result
    
    main_key = 'error_ratio'
    main = store[main_key]
    main.columns = [main_key]
    for key, series in store.items():
        if key == main_key:
            continue
        series.columns = [key]
        main = main.join(series)
    return main

if __name__ == "__main__":
    dir_list = ['2021_12_25','2021_12_26','2021_12_27']
    pd_list = []
    for dir_name in dir_list:
        pd_list.append(get_data_from_dir(dir_name))
    total_data = pd.concat(pd_list)
    total_data.to_csv("total.csv")