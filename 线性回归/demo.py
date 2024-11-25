import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score

# 设置随机数的种子，保证每次运行程序，生成的随机数相同.
np.random.seed(42)

# 生成样本点
start = 10
m = 20
x = np.arange(start=start, stop=start + m)
y = 1.9 * x + 0.7 + np.random.randn(m)  # 加入一些噪声
y = np.round(y, 2)  # 保留2位小数.

# 输出样本点.
for i in range(m):
    print(f"x{i + 1}: {x[i]}, y{i + 1}: {y[i]}")

x_bar = np.average(x)  # 自变量平均值.
y_bar = np.average(y)  # 因变量平均值.
var_x = np.sum((x - x_bar) ** 2) / (m - 1)  # 自变量的方差.
cov_x_y = np.sum((x - x_bar) * (y - y_bar)) / (m - 1)  # 自变量与因变量的协方差.
theta_1 = cov_x_y / var_x  # 计算未知参数theta_1.
theta_0 = y_bar - theta_1 * x_bar  # 计算未知参数theta_0.
print(f'x_bar={x_bar}, y_bar={y_bar}, var_x={var_x}, cov_x_y={cov_x_y}, theta_1={theta_1}, theta_0={theta_0}')

# 线性模型表达式.
y_hat = theta_1 * x + theta_0
print('y_hat_1=', y_hat[0])  # 预测第一个样本的生成成本.
# 预测新样本的生产成本.
x_new = 30
y_hat_new = theta_1 * x_new + theta_0
print('y_hat_new=', y_hat_new)

# 自定义实现评估指标.
mse = np.average((y_hat - y) ** 2)  # 均方误差.
mae = np.average(abs(y_hat - y))  # 平均绝对值误差
rmse = np.sqrt(mse)  # 均方根误差
r2 = np.sum((y_hat - y_bar) ** 2) / np.sum((y - y_bar) ** 2)  # 决定系数.
print(f'mse={mse}, mae={mae}, rmse={rmse}, r2={r2}')

# 使用sklearn内置模型训练，并输出评估指标.
model = LinearRegression()
model.fit(x.reshape(-1, 1), y)
model_y_hat = model.predict(x.reshape(-1, 1))
model_mse = mean_squared_error(y, model_y_hat)
model_mae = mean_absolute_error(y, model_y_hat)
model_rmse = root_mean_squared_error(y, model_y_hat)
model_r2 = r2_score(y, model_y_hat)
print(f'model_mse={model_mse}, model_mae={model_mae}, model_rmse={model_rmse}, model_r2={model_r2}')
print(f'model_coef={model.coef_}, model_intercept={model.intercept_}')

# 绘制散点图
plt.rcParams['font.sans-serif'] = ['SimHei']  # 避免散点图显示中文乱码.
plt.scatter(x, y)
plt.plot(x, y_hat, color='red')
# 添加标题和坐标轴标签
plt.title("产品个数-生产成本散点图")
plt.xlabel("产品个数")
plt.ylabel("生产成本")

# 显示图形
plt.show()
