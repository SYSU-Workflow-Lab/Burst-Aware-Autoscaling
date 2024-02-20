from logging import error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
# from pl_predictor import LightningPredictor
# import pytorch_lightning as pl

import os
from joblib import dump, load

from util.constdef import read_profilng_data
from predictor_network import PerformancePredictor, CustomDataset

# 包含以下几个模块：
# 数据整合
def find_platform(pd_data, index=0, 最大流量偏差值=12, 最大实例偏差值=0, 最小平台允许点数=8):
    """
        要求找到流量和实例数都平稳的平台，返回平台的开始点和结束点
    Args:
        pd_data: pandas格式的数据
        index: 开始的位置
        bias: 能够接受的流量变动范围
        min_num: 构成平台的最小数量，取2。不能比4大，最大应该是3.
    Returns:
        返回index之后的一个平台（含index）
        (start,end) 平台的开始点和结束点
        如果本次探索，则start=-1,end = 新的index建议值
        如果探索失败（后续没有），则两者都是-1
    """
    # 具体实现方法
    # 从当前点出发，搜索未来的8个点，取diff
    allow_bias = 最小平台允许点数 * 3
    future_workload_nested = pd_data['workload'].values[index:index+allow_bias]
    diff_future_workload_nested = np.abs(np.diff(future_workload_nested))
    # 理想情况：在这8个点中
    legal_point_list = np.where(diff_future_workload_nested < 最大流量偏差值)[0] + index
    if len(legal_point_list) < 最小平台允许点数:
        # 如果合法的点数没有达到最小平台允许点数，则说明后面不可能具备足够多的数据，直接退出搜索
        if index + allow_bias < len(pd_data):
            next_index = index + allow_bias
            if len(legal_point_list) > 0:
                next_index = legal_point_list[0] + 1
            return (-1, next_index)
            # return find_platform(pd_data, index = next_index, 最大流量偏差值=最大流量偏差值, 最小平台允许点数=最小平台允许点数, 最大实例偏差值=最大实例偏差值)
        else:
            return (-1, -1)
    platform_start_point = legal_point_list[0]
    workload_data = pd_data['workload'].values
    instance_data = pd_data['instance'].values
    pivot = platform_start_point
    while pivot < len(pd_data):
        if pivot + 1 == len(pd_data):
            break
        if np.abs(instance_data[pivot] - instance_data[pivot+1]) > 最大实例偏差值 or \
        np.abs(workload_data[pivot] - workload_data[pivot+1]) > 最大流量偏差值:
            break
        pivot += 1
    platform_end_point = pivot
    # 判断长度是否足够
    if platform_end_point - platform_start_point < 最小平台允许点数:
        return (-1, platform_end_point+1)
        # return find_platform(pd_data, index = platform_end_point+1, 最大流量偏差值=最大流量偏差值, 最小平台允许点数=最小平台允许点数, 最大实例偏差值=最大实例偏差值)
    else:
        return (platform_start_point, platform_end_point)
    

# 给定文件名称，读取指定文件中的内容
def filter_data(pd_data):
    """
        过滤数据，原因在于在流量刚开始变换的时候会非常不稳定，产生无效的数据影响到预测的效率
        目前的想法是，直接抛弃掉这部分数据
        原理：
            从最开始搜索流量平台，流量平台的定义是有三个或以上的流量距离较近（10以下）
            从当前平台到下一个平台之间，如果有1~2个点的变动比较异常，则直接将其舍弃掉
        技术问题：
        1. 如何识别流量平台
        2. 如何舍弃数据
        3. 如何选择哪些数据舍弃
        Args:
            pd_data:
            第一个是25日
            第二个是26日
            第三个是27日
    """
    index = 0
    rest_array = np.zeros(len(pd_data)) # 抛弃的实现：将要保留的部分置为1
    最小平台允许点数 = 2
    # print(f"当前最小平台允许点数{最小平台允许点数}")
    while index < len(pd_data):
        start,end = find_platform(pd_data, index=index, 最小平台允许点数=最小平台允许点数)
        # print(start,end)
        if start == -1 and end == -1:
            # 直接把后面的数据全部抛弃
            break
        elif start == -1:
            index = end
        else:
            rest_array[start:end] = np.ones(end-start)
            # 获取下一个平台的流量数据和实例数据
            # 抛弃两个平台之间的所有数据
            # 更新prev_index等只
            index = end+1
    return pd_data.iloc[np.where(rest_array!=0)]

# 给定pd数组，拼接内容，清洗非nan数据，返回指定数列的数据（字典形式）
def extract_pd_data(pd_data_list, col_list = None):
    # 使用平台来去除转换阶段的数据
    new_data_list = []
    for data in pd_data_list:
        # new_data_list.append(filter_data(data))
        new_data_list.append(data)
    total_data = pd.concat(new_data_list)

    # total_data = pd.concat(pd_data_list)
    # 如果col_list = None，则返回所有列
    if col_list == None:
        col_list = total_data.columns.values

    aim_data = total_data[col_list]
    not_null_sel = np.where(aim_data.isna().sum(axis=1).values == 0)
    not_null_data = aim_data.iloc[not_null_sel]
    result_dict = {}
    for col in not_null_data.columns.values:
        result_dict[col] = not_null_data[col].values
    # NOTE 制作额外的输入输出以和之前的相对应
    result_dict['workload'] = result_dict['wkl']
    result_dict['cpu_utilization'] = result_dict['cpu']
    result_dict['instance'] = result_dict['ins']
    result_dict['response_avg_istio'] = result_dict['res']
    result_dict['error_ratio'] = result_dict['err']
    return result_dict

# 对于给定数据，去除outlier
from sklearn.ensemble import IsolationForest
def detect_and_clean_outliers(trainX, trainY, data_dict):
    acc = data_dict['error_ratio'] < 1.0
    trainX = trainX[acc]
    trainY = trainY[acc]

    if len(trainX.shape)>1:
        data_before_detect = np.concatenate([trainX[:,0].reshape(-1,1), trainY.reshape(-1,1)], axis=1)
    else:
        data_before_detect = np.concatenate([trainX.reshape(-1,1), trainY.reshape(-1,1)], axis=1)
    # data_before_detect = np.concatenate([trainX, trainY.reshape(-1,1)], axis=1)
    # 确定，给定的两个数组都是一维的
    # clf = IsolationForest(n_estimators=20, warm_start=True)
    clf = LocalOutlierFactor(n_neighbors=10)
    lof_result = clf.fit_predict(data_before_detect)
    
    inlier = lof_result == 1
    outlier = lof_result==-1
    return trainX[inlier], trainY[inlier], inlier
    
# 训练预测器SVR，并画图
def train_SVR(trainX, trainY):
    model = SVR(kernel="rbf", C=100)
    if len(trainX.shape) == 1:
        trainX = trainX.reshape(-1,1)
    kf = KFold(n_splits=10,shuffle=True)
    total = 0
    for i in range(10):
        loss = 0
        count = 0
        for train_idx, val_idx in kf.split(trainX):
            model.fit(trainX[train_idx], trainY[train_idx])
            predict_Y = model.predict(trainX[val_idx])
            loss += mean_squared_error(trainY[val_idx], predict_Y)
            count += 1
        total += loss / count
    print(total/10)

    testX = np.arange(min(trainX[:,0]),max(trainX[:,0]),0.2)
    testX = testX.reshape(-1,1)
    testY = model.predict(testX)
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
    plt.rcParams.update({'font.size': 20})
    plt.rcParams["figure.figsize"] = (8,6)
    plt.scatter(trainX,trainY,color=((228/255,26/255,28/255)),label='训练数据',alpha=0.5)
    plt.plot(testX,testY,color=((55/255,126/255,184/255)),label='预测结果',linewidth=5)
    plt.xlabel("单位容器内流量值")
    plt.legend(loc=2)
    plt.ylabel("响应时间(ms)") # 响应时间(ms)
    plt.savefig("res_predict.pdf")
    # plt.ylabel("CPU占用率(%)")
    # plt.savefig("cpu_predict.pdf")
    plt.show()
    return model

from sklearn.neighbors import KNeighborsRegressor
def train_KNN(trainX,trainY):
    model = KNeighborsRegressor()
    if len(trainX.shape) == 1:
        trainX = trainX.reshape(-1,1)
    kf = KFold(n_splits=10,shuffle=True)
    total = 0
    for i in range(10):
        for train_idx, val_idx in kf.split(trainX):
            model.fit(trainX[train_idx], trainY[train_idx])
            predict_Y = model.predict(trainX[val_idx])
            total += mean_squared_error(trainY[val_idx], predict_Y)
    print(total/10/10)
    # testX = np.arange(min(trainX),max(trainX),0.2)
    # testY = model.predict(testX.reshape(-1,1))
    # plt.scatter(testX,testY,color='b')
    # plt.scatter(trainX,trainY,color='r')
    # plt.show()
    return model

from sklearn.linear_model import HuberRegressor
def train_Huber(trainX,trainY):
    model = HuberRegressor()
    if len(trainX.shape) == 1:
        trainX = trainX.reshape(-1,1)
    kf = KFold(n_splits=10,shuffle=True)
    total = 0
    for i in range(10):
        for train_idx, val_idx in kf.split(trainX):
            model.fit(trainX[train_idx], trainY[train_idx])
            predict_Y = model.predict(trainX[val_idx])
            total += mean_squared_error(trainY[val_idx], predict_Y)
    print(total/10/10)
    # testX = np.arange(min(trainX),max(trainX),0.2)
    # testY = model.predict(testX.reshape(-1,1))
    # plt.scatter(testX,testY,color='b')
    # plt.scatter(trainX,trainY,color='r')
    # plt.show()
    return model

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold
def train_DTR(trainX, trainY):
    model = DecisionTreeRegressor()
    if len(trainX.shape) == 1:
        trainX = trainX.reshape(-1,1)
    kf = KFold(n_splits=10,shuffle=True)
    total = 0
    for i in range(10):
        for train_idx, val_idx in kf.split(trainX):
            model.fit(trainX[train_idx], trainY[train_idx])
            predict_Y = model.predict(trainX[val_idx])
            total += mean_squared_error(trainY[val_idx], predict_Y)
    print(total/10/10)
    # if trainX.shape[-1] == 2:
    #     testX = np.arange(min(trainX[:,0]),max(trainX[:,0]),0.2)
    #     testY = model.predict(testX.reshape(-1,1))
    # else:
    #     testX = np.arange(min(trainX),max(trainX),0.2)
    #     testY = model.predict(testX.reshape(-1,1))
    # plt.scatter(testX,testY,color='b')
    # plt.scatter(trainX,trainY,color='r')
    # plt.show()
    return model

def train_RF(trainX, trainY):
    model = RandomForestRegressor()
    if len(trainX.shape) == 1:
        trainX = trainX.reshape(-1,1)
    kf = KFold(n_splits=10,shuffle=True)
    total = 0
    for i in range(10):
        for train_idx, val_idx in kf.split(trainX):
            model.fit(trainX[train_idx], trainY[train_idx])
            predict_Y = model.predict(trainX[val_idx])
            total += mean_squared_error(trainY[val_idx], predict_Y)
    print(total/10/10)
    return model

import lightgbm as lgbm
def train_LGBM(trainX, trainY):
    model = lgbm.LGBMRegressor()
    if len(trainX.shape) == 1:
        trainX = trainX.reshape(-1,1)
    kf = KFold(n_splits=10,shuffle=True)
    total = 0
    for i in range(10):
        for train_idx, val_idx in kf.split(trainX):
            model.fit(trainX[train_idx], trainY[train_idx])
            predict_Y = model.predict(trainX[val_idx])
            total += mean_squared_error(trainY[val_idx], predict_Y)
    print(total/10/10)
    return model

from tqdm import tqdm
def train_GP(trainX, trainY):
    # 效果一般
    # kernel = WhiteKernel(noise_level=1,noise_level_bounds=(1e-10, 1e3)) \
    #             + RBF(length_scale=1, length_scale_bounds=(1e-5, 1e3))
    model = GaussianProcessRegressor()
    if len(trainX.shape) == 1:
        trainX = trainX.reshape(-1,1)
    kf = KFold(n_splits=10,shuffle=True)
    total = 0
    for train_idx, val_idx in tqdm(kf.split(trainX)):
        model.fit(trainX[train_idx], trainY[train_idx])
        predict_Y = model.predict(trainX[val_idx])
        total += mean_squared_error(trainY[val_idx], predict_Y)
    print(total/10)
    return model

from tqdm import trange
def train_with_NN(data_dict, epoch_num = 100,aim='cpu_utilization'):
    total = 0
    for i in trange(10):
        network = PerformancePredictor(enc_number = 1, n_output = 1).double()
        dataset = CustomDataset(data_dict,aim=aim)
        data_loader = DataLoader(dataset,batch_size=32,shuffle=True,drop_last=False)
        criterion = nn.MSELoss()
        model_optimizer = optim.Adam(network.parameters(), lr=1e-2)
        network.train()
        for epoch in range(epoch_num):
            loss_list = []
            for data,trainY in data_loader:
                trainX = data.unsqueeze(dim=-1)
                # trainX = torch.transpose(torch.stack(data), 0, 1)
                # assert trainX.shape[0] == len(trainY)

                trainY = trainY.unsqueeze(dim=-1)
                model_optimizer.zero_grad()

                predictY = network(trainX)
                loss = criterion(trainY, predictY)
                loss.backward()
                loss_list.append(loss.item())
                model_optimizer.step()
                
            # print(np.mean(loss_list))
        trainX = torch.Tensor(data_dict['per_workload'][dataset.error < 1.0][dataset.test_data]).double().unsqueeze(-1)
        trainX = (trainX - dataset.x_mean) / dataset.x_std
        trainY = network(trainX)
        trainY = trainY * dataset.y_std + dataset.y_mean
        actY = data_dict[aim][dataset.error < 1.0][dataset.test_data]
        total += mean_squared_error(trainY.detach().numpy().ravel(),actY)
    print(f"{aim}: loss is {total/10}")
    return network    

# def train_with_PL(data_dict, epoch_num = 10):
#     model = LightningPredictor(enc_number = 3, n_output = 1)
#     dataset = CustomDataset(data_dict)
#     data_loader = DataLoader(dataset,batch_size=32,shuffle=True,drop_last=False)
#     trainer = pl.Trainer(max_epochs = 100)
#     trainer.fit(model, data_loader)
    
    return model   

from sklearn.ensemble import GradientBoostingRegressor
def train_with_GBR(trainX, trainY):
    # 效果一般
    # kernel = WhiteKernel(noise_level=1,noise_level_bounds=(1e-10, 1e3)) \
    #             + RBF(length_scale=1, length_scale_bounds=(1e-5, 1e3))
    model = GradientBoostingRegressor()
    if len(trainX.shape) == 1:
        trainX = trainX.reshape(-1,1)
    kf = KFold(n_splits=10,shuffle=True)
    total = 0
    for i in range(10):
        for train_idx, val_idx in tqdm(kf.split(trainX)):
            model.fit(trainX[train_idx], trainY[train_idx])
            predict_Y = model.predict(trainX[val_idx])
            total += mean_squared_error(trainY[val_idx], predict_Y)
    print(total/10/10)
    return model

def train_with_MLP(trainX, trainY):
    # 效果一般
    # kernel = WhiteKernel(noise_level=1,noise_level_bounds=(1e-10, 1e3)) \
    #             + RBF(length_scale=1, length_scale_bounds=(1e-5, 1e3))
    model = MLPRegressor(activation='logistic',max_iter=2000,early_stopping=True)
    if len(trainX.shape) == 1:
        trainX = trainX.reshape(-1,1)
    kf = KFold(n_splits=10,shuffle=True)
    total = 0
    for i in range(10):
        for train_idx, val_idx in tqdm(kf.split(trainX)):
            model.fit(trainX[train_idx], trainY[train_idx])
            predict_Y = model.predict(trainX[val_idx])
            total += mean_squared_error(trainY[val_idx], predict_Y)
    print(total/10/10)
    return model


def print_data_with_model(model):
    expect_data = np.arange(1,250,0.2)
    output_data = []
    for workload in expect_data:
        data = torch.Tensor([workload, 2.0, 0.0]).unsqueeze(dim=0).double()
        out = model(data)
        output_data.append(out.item())
    plt.scatter(expect_data,output_data)
    plt.show()

def read_file_list():
    # file_list = ['anomaly.csv', 'exp1.csv', 'exp2.csv', 'result.csv'] # 之前2000的数据
    # file_list = ['2021_12_25.csv', '2021_12_26.csv', '2021_12_27.csv'] # 最近10000的数据
    # file_list = ['total.csv'] # 整合后的10s级别数据
    # file_list = ['2021_12_29.csv','2021_12_30.csv','2021_12_31.csv']
    file_list = ['26_08_01.csv']
    pd_data_list = []
    for file_name in file_list:
        pd_data_list.append(read_profilng_data(file_name=file_name))
    return pd_data_list

# 训练两个SVM预测器并进行存储在predictor_model文件夹内
def train_ML_model_and_store(model_name):
    pd_data_list = read_file_list()
    data_dict = extract_pd_data(pd_data_list)
    data_dict['per_workload'] = data_dict['workload'] / data_dict['instance']
    per_workload = data_dict['workload'] / data_dict['instance']
    cpu_utilization = data_dict['cpu_utilization']
    response_time = data_dict['response_avg_istio']

    # trainX = per_workload
    # trainY = cpu_utilization
    # trainX, trainY, _ = detect_and_clean_outliers(per_workload, cpu_utilization, data_dict)
    # model_name = "SVR"
    # if model_name == "SVR":
    #     cpu_model = train_SVR(trainX, trainY)
    # elif model_name == "DTR":
    #     cpu_model = train_DTR(trainX, trainY)
    # elif model_name == "KNN":
    #     cpu_model = train_KNN(trainX, trainY)
    # elif model_name == "GP":
    #     cpu_model = train_GP(trainX, trainY)
    # elif model_name == "RF":
    #     cpu_model = train_RF(trainX, trainY)
    # elif model_name == "LGBM":
    #     cpu_model = train_LGBM(trainX,trainY)
    # elif model_name == "huber":
    #     cpu_model = train_Huber(trainX,trainY)
    # elif model_name == "GBR":
    #     cpu_model = train_with_GBR(trainX,trainY)
    # elif model_name == "MLP":
    #     cpu_model = train_with_MLP(trainX,trainY)
    
    # dump(cpu_model, os.path.join('data','model','cpu_predictor.joblib'))

    trainX, trainY = per_workload, response_time
    # trainX = np.concatenate([data_dict['per_workload'].reshape(-1,1), data_dict['instance'].reshape(-1,1)],axis=1)
    trainX, trainY, _ = detect_and_clean_outliers(trainX, trainY, data_dict)
    # model_name = "GBR"
    if model_name == "SVR":
        res_model = train_SVR(trainX, trainY)
    elif model_name == "DTR":
        res_model = train_DTR(trainX, trainY)
    elif model_name == "KNN":
        res_model = train_KNN(trainX, trainY)
    elif model_name == "GP":
        res_model = train_GP(trainX, trainY)
    elif model_name == "RF":
        res_model = train_RF(trainX, trainY)
    elif model_name == "LGBM":
        res_model = train_LGBM(trainX,trainY)
    elif model_name == "huber":
        res_model = train_Huber(trainX,trainY)
    elif model_name == "GBR":
        res_model = train_with_GBR(trainX,trainY)
    elif model_name == "MLP":
        res_model = train_with_MLP(trainX,trainY)
    # dump(res_model, os.path.join('data','model','res_predictor.joblib'))

# 想办法打印一下响应时间预测模型
# per_workload = np.arange(0.2,27.5,0.01)
# instance = np.ones_like(per_workload)
# store = []
# for i in range(1,50):
#     trainx = np.stack([per_workload,instance*i]).transpose()
#     store.append(res_model.predict(trainx))

# plt.scatter(trainX[:,0],trainY)
# plt.plot(per_workload,store[5],color='r')
# plt.plot(per_workload,store[15],color='g')
# plt.plot(per_workload,store[25],color='b')
# plt.show()

def train_NN_model_and_store():
    pd_data_list = read_file_list()
    data_dict = extract_pd_data(pd_data_list)
    data_dict['per_workload'] = data_dict['workload'] / data_dict['instance']
    per_workload = data_dict['workload'] / data_dict['instance']
    cpu_utilization = data_dict['cpu_utilization']
    response_time = data_dict['response_avg_istio']
    print("train CPU")
    train_with_NN(data_dict,aim="cpu_utilization")
    print("train response time")
    train_with_NN(data_dict,aim="response_avg_istio")

# NOTE 3d图
def draw_3d_pic():
    pd_data_list = read_file_list()
    data_dict = extract_pd_data(pd_data_list)
    data_dict['per_workload'] = data_dict['workload'] / data_dict['instance']
    per_workload = data_dict['per_workload']
    cpu_utilization = data_dict['cpu_utilization']
    response_time = data_dict['response_avg_istio']
    error_ratio = data_dict['error_ratio']
    inlier = error_ratio<1
    ax = plt.subplot(111,projection='3d')
    ax.scatter(per_workload[inlier],response_time[inlier],error_ratio[inlier])
    ax.set_zlabel('error')
    ax.set_ylabel('res')
    ax.set_xlabel('workload')
    plt.show()

# 主函数功能部分
# 用于进一步的测试
def main():
    # 1. 读取指定的文件内容
    pd_data_list = read_file_list()
    data_dict = extract_pd_data(pd_data_list)
    # model = train_with_PL(data_dict)
    # print_data_with_model(model)
    # 计算单位流量作为X
    trainX = data_dict['workload'] / data_dict['instance']
    trainY = data_dict['cpu_utilization']
    trainX, trainY, outlier_list = detect_and_clean_outliers(trainX, trainY, data_dict)
    # 使用RF
    inst = data_dict['instance'] <= 2
    erro = data_dict['error_ratio'] > 1
    result = np.logical_or(inst,erro)
    train_SVR(trainX, trainY)
    # 基于CPU占用确定对应的单位实例数，进行分段。
    # 分段大体对应90%左右的流量，在190左右。前后分别使用不同的预测器进行预测
    # first_part = trainX<190
    # second_part = trainX>=190
    # trainX, trainY = trainX[first_part], trainY[first_part]

    # 导入到对应的函数中进行预测，产生预测结果图以及偏差曲线。

def train_3d_model(data_dict,x="workload",y="instance",z="cpu_utilization"):
    ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
    #  将数据点分成三部分画，在颜色上有区分度
    ax.scatter(data_dict[x],data_dict[y],data_dict[z])  # 绘制数据点
    ax.set_zlabel(z)  # 坐标轴
    ax.set_ylabel(y)
    ax.set_xlabel(x)
    plt.show()

def validate_current_cluster_state():
    file_list = ['28_12_26.csv']
    pd_data_list = []
    for file_name in file_list:
        pd_data_list.append(read_profilng_data(file_name=file_name))
    data_dict = extract_pd_data(pd_data_list)
    res_model = load(os.path.join('data','model','res_predictor.joblib'))
    data_dict['per_workload'] = data_dict['workload'] / data_dict['instance']
    trainX = data_dict['per_workload']
    testX = np.arange(min(trainX),max(trainX),0.2)
    testX = testX.reshape(-1,1)
    testY = res_model.predict(testX)
    plt.scatter(data_dict['per_workload'], data_dict['res'])
    plt.plot(testX,testY,color='b')
    plt.show()


if __name__ == "__main__":
    # 下列方法，输入指定的数据，输出以MSE规定的方差。
    # 其中，CPU部分输入为per_workload，输出为CPU占用率
    # 响应时间部分，输入per_workload,instance，输出为响应时间
    # validate_current_cluster_state()
    train_ML_model_and_store(model_name="SVR")
    # train_NN_model_and_store()


# x = "per_workload"
# y = "total_cpu_usage"
# z = "cpu_utilization"
# k1 = np.logical_and(data_dict["instance"]==5, data_dict["cpu_utilization"]>90)
# ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
# #  将数据点分成三部分画，在颜色上有区分度
# # ax.scatter(data_dict[x],data_dict[y],data_dict[z],color='b')  # 绘制数据点
# ax.scatter(data_dict[x][k1],data_dict[y][k1],data_dict[z][k1],color='r')  # 绘制数据点
# ax.set_zlabel(z)  # 坐标轴
# ax.set_ylabel(y)
# ax.set_xlabel(x)
# plt.show()

# k1 = data_dict["instance"] == 1
# plt.scatter(data_dict[x][k1],data_dict[z][k1])
# plt.show()