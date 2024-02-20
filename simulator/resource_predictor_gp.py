import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 资源预测器

# def read_data_from_csv(pd_file):
pd_file = "exp2.csv"
df = pd.read_csv(pd_file)
not_null_sel = np.where(df.isna().sum(axis=1).values == 0)
workload = df['workload'].values[not_null_sel]
instance_num = df['instance'].values[not_null_sel]
per_workload = workload / instance_num
response_avg = df['response_avg'].values[not_null_sel]
response_istio = df['response_avg_istio'].values[not_null_sel]
cpu_utilization = df['cpu_utilization'].values[not_null_sel]
P95_istio = df['P95_istio'].values[not_null_sel]
P95_custom = df['P95_custom'].values[not_null_sel]
error_ratio = df['error_ratio'].values[not_null_sel]
# 参考：移除nan值的数据
# 

# 散点图
# 流量
plt.scatter(per_workload, P95_custom,color='r')
plt.scatter(per_workload, response_avg,color='b')
plt.xlim(0,200)
plt.ylim(0,5)
plt.show()

# CPU 占用率
plt.scatter(per_workload, cpu_utilization)
plt.show()

# SVR

# 高斯回归
# 相关函数
X = per_workload
y = P95_istio
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
l = 0.1
sigma_f = 100
kernel = ConstantKernel(constant_value=sigma_f,constant_value_bounds=(1e-3, 1e3)) \
            * RBF(length_scale=l, length_scale_bounds=(1e-3, 1e3))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
X = X.reshape(-1,1)
gp.fit(X, y)
X_star = np.linspace(1,300,600).reshape(-1,1)
y_pred = gp.predict(X_star)
y_pred, std = gp.predict(X_star, return_std=True)

plt.plot(X_star.ravel(), y_pred)
plt.fill_between(x = X_star.ravel(), y1 = y_pred-1.95*std, y2 = y_pred + 1.95*std, color='green', alpha=0.3, label='Credible Interval')
plt.show()