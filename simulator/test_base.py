from basic_env.predict_env import PredictBasicEnv
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from pmdarima import model_selection
import numpy as np
from tqdm import trange
from util.metric import QuantileLoss

def burst_validator(workload_name ,use_burst=True):
    env = PredictBasicEnv(workload_name=workload_name, use_burst=use_burst)
    burst_detector = env.burst_detector
    burst_array = burst_detector.burst_array
    train_arma_model_and_predict(env.train_workload_data[:500],env.train_workload_data[500:600])

def train_arma_model_and_predict(train_data, test_data, max_length=168*3):
    """
        ARMA的预测器，直接使用训练段跑数据看效果
        正则化：可以先取对数再做正则化，或者不进行正则化（可能有问题）
        Args:
            train_data为在此之前的non-burst数据
            test_data为当前的burst数据，滚动更新
    """
    # 归一化处理。取数据的P90 - P10做归一化
    aim_data = np.sort(train_data)
    aim_max, aim_min = aim_data[int(0.9 * len(aim_data))], aim_data[int(0.1 * len(aim_data))]
    train_data = (train_data - aim_min) / (aim_max - aim_min)
    test_data = (test_data - aim_min) / (aim_max - aim_min)
    if len(train_data) > max_length:
        train_data = train_data[-max_length:]
    model = pm.auto_arima(train_data, start_p=1, start_q=1, d=0, max_p=4, max_q=4)
    # 根据train部分预测的质量来进行分析
    insample_predict_data, conf_int = model.predict_in_sample(start=0,end=len(train_data)-1,
                                                    return_conf_int=True)
    loss = QuantileLoss(quantiles=[0.1,0.5,0.9])
    qrisk = loss.numpy_normalised_quantile_loss(np.stack([conf_int[:,0], insample_predict_data, conf_int[:,1]]).transpose(), train_data, quantile=0.9)
    result = np.zeros_like(test_data)
    for i in trange(len(test_data)):
        # preds = model.predict(n_periods=1)
        preds, conf_int = model.predict(n_periods=1, alpha=0.1, # 返回95%置信区间的结果
                                return_conf_int=True)
        result[i] = conf_int[0,1].item()
        model.update(test_data[i])
    return result * (aim_max - aim_min) + aim_min


def burst_value_extractor(train_step=10000,seq_len=4):
    """
        其他类型的预测器，比如SVM等
        将burst部分的数据打包成训练数据+label，其中label部分必须全部为burst部分数据
        正则化方法采用本地正则化方法
        返回的结果为训练数据、pivot数组（供反正则化使用）
    """

# def train_burst_predictor(seq_len=4)