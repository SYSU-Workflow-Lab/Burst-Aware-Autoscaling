# 仓库介绍

本文件夹主要负责的是模拟器的构建，特别是gym-style的环境构建。

其中涉及到了资源预测器的产生与组合

注意：本系统必须在simulator目录下才能运行

## 文件介绍

### 入口文件

* `ppo_agent.py`，PPO等组件的核心入口文件
* `data_analysis,ipynb`，数据分析

### 其他文件

* `resource_predictor.py`，产生基于SVM的资源预测并进行保存。
  * 部分时候也充当了其他测试函数的main函数
* `/basic_env`，标准环境构建
  * `basic_env.py`，gym-style环境的构建，基于两个SVM预测器进行
  * `reward_func.py`，设计好的奖励函数
* `/data`，数据文件，包括Profiling的结果数据，以及源自维基百科的模拟流量数据
  * `/longterm`，长期预测器的文件，产生文件为informer.py下的adp_test方法
  ```python
  np.save(os.path.join(folder_path, 'pred.npy'), preds)
  np.save(os.path.join(folder_path, 'true.npy'), trues)
  np.save(os.path.join(folder_path, 'origin.npy'), origins) # 输入部分的数据
  np.save(os.path.join(folder_path, 'origin_data.npy'), origin_true_data) # 输入部分的数据
  np.save(os.path.join(folder_path, 'origin_timestamp.npy'), timestamp_data) # 输入部分的数据
  ```
  * `/profiling`，进行随机测试的文件，资源预测器的数据来源。
  * `/workload`，流量数据，csv文件
* `/model`，SVM预测器的存放地点，包括`cpu_predictor.joblib`和`res_predictor.joblib`

### 实验文件

* `resource_predictor_gp.py`，使用高斯过程回归
* `pl_predictor.py`，使用pytorch-lightning
* `predictor_network.py`，使用普通ANN进行预测

## 开源代码使用

PPO部分参考了小雅的单文件实现