# 1. 需要根据最新的部分，改正a-sarsa的环境部分
# 2. 
import logging
from ppo.common import train_and_evaluate
from baseline_method.dqn.run import train_sarima_model, validate_asarima_model
from util.constdef import TEST_MODE, TRAIN_MODE, VALI_MODE

# 训练一个算法
def asarsa_trainer(dataStore, workload_name, # 此部分超参数，参考/basic_env/predict_env.py中的PredictBasicEnv:__init__的设置
    train_step=10000, vali_step=3000, test_step=3000, 
    action_max=5, sla=8., des_mean=10000, des_std=3000,
    use_burst=False, is_burst=False):
    """
        对比算法复现: A-SARSA
        Args:
            dataStore，记录用的pd.DataFrame
    """
    logging.info(f"asarsa prediction method for {workload_name}")
    train_sarima_model(workload_name, action_max=action_max,
        train_step=train_step, vali_step=vali_step, test_step=test_step,
        use_burst=use_burst, is_burst=is_burst,sla=sla,des_mean=des_mean, des_std=des_std)
    # 记录验证部分数据
    return dataStore

# 对算法训练出来的模型进行验证
def asarsa_validator(dataStore, workload_name, # 此部分超参数，参考/basic_env/predict_env.py中的PredictBasicEnv:__init__的设置
    train_step=10000, vali_step=3000, test_step=3000, 
    action_max=5, sla=8., des_mean=10000, des_std=3000,
    use_burst=False, is_burst=False):
    if use_burst:
        if is_burst:
            train_burst = 2
        else:
            train_burst = 1
    else:
        train_burst = 0

    mode_list = {TRAIN_MODE,VALI_MODE,TEST_MODE}
    for mode in mode_list:
        # in_burst表示使用non-burst-aware的模型
        dataStore = validate_asarima_model(dataStore, workload_name,cur_mode=mode,action_max=action_max,
            train_step=train_step, vali_step=vali_step, test_step=test_step,
            in_burst=False,train_burst=train_burst,sla=sla, des_mean=des_mean, des_std=des_std)
    return dataStore