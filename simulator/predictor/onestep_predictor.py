# 单步预测器。

from util.constdef import read_onestep_prediction_result, parse_workload_data

class OneStepPredictor:
    def __init__(self, workload_name):
        """
            根据输入的流量名称读取单步预测器
        """
        origin_data, origin_timestamp, origin, pred, true = read_onestep_prediction_result(workload_name)
        self.origin_data = origin_data
        self.origin_timestamp = origin_timestamp
        self.pred = pred
        self.true = true
        self.origin_pred = pred
        self.origin_true = true
    
    def normalize_predictor(self, aim_mean, aim_std, des_mean, des_std):
        self.pred = parse_workload_data(self.pred, aim_mean=aim_mean, aim_std=aim_std, des_mean=des_mean, des_std=des_std)
        # self.true = parse_workload_data(self.true, aim_mean=aim_mean, aim_std=aim_std, des_mean=des_mean, des_std=des_std)
