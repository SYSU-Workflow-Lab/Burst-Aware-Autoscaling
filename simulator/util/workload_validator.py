from basic_env.predict_env import PredictBasicEnv
from util.metric import SMAPE
def onestep_workload_validator(dataStore, workload_name, action_max=10,
        train_step=6000, vali_step=4000, test_step=4000,
        use_burst=False, is_burst=False,sla=35.,des_mean=500, des_std=125):
    env = PredictBasicEnv(workload_name=workload_name, action_max=action_max,
        train_step=train_step, vali_step=vali_step, test_step=test_step,
        use_burst=use_burst, is_burst_bool=is_burst,sla=sla,des_mean=des_mean, des_std=des_std)
    # TODO 这个是不准确的
    total_pred_value = env.onestep_predictor.origin_pred[env.start_point+1:env.start_point+env.train_step+env.vali_step+env.test_step+1].ravel()
    total_true_value = env.onestep_predictor.origin_true[env.start_point+1:env.start_point+env.train_step+env.vali_step+env.test_step+1].ravel()
    # 计算MSE等值，存放在dataStore中
    smape_value = SMAPE(total_pred_value, total_true_value)
    dataStore = dataStore.append({
        "workload_name": workload_name,# 流量名称
        "addition_info2": smape_value,
    }, ignore_index=True)
    return dataStore