# 长期预测器。说是预期器，实际上是读取预测器信息并进行返回
from util.constdef import read_longterm_prediction_result, parse_workload_data, LONGTERM_WORKLOAD_DIR

class LongtermPredictor:
    def __init__(self, workload_name,workload_dir=LONGTERM_WORKLOAD_DIR):
        """
            根据输入的流量名称读取单步预测器
        """
        origin_data, origin_timestamp, origin, pred, true = read_longterm_prediction_result(workload_name,workload_dir)
        self.origin_data = origin_data
        self.origin_timestamp = origin_timestamp
        self.pred = pred
        self.true = true
    
    def normalize_predictor(self, aim_mean, aim_std, des_mean, des_std):
        self.pred = parse_workload_data(self.pred, aim_mean=aim_mean, aim_std=aim_std, des_mean=des_mean, des_std=des_std)