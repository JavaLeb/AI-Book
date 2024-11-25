import numpy as np
import matplotlib.pyplot as plt

# 设置随机数的种子，保证每次运行程序，生成的随机数相同.
np.random.seed(42)

# 生成样本点
start = 10
num = 20
x = np.arange(start=start, stop=start + num)
y = 1.9 * x + 0.7 + np.random.randn(num)  # 加入一些噪声
y = [round(i, 2) for i in y]  # 保留2位小数.

# 输出样本点.
for i in range(num):
    print(f"x: {x[i]}, y: {y[i]}")

x_bar = np.average(x)
y_bar = np.average(y)
var_x = np.var(x)
cov_x_y = np.cov(x, y)[0, 1]
theta_1 = cov_x_y / var_x
theta_0 = y_bar - theta_1 * x_bar
print(f'x_bar={x_bar}, y_bar={y_bar}, var_x={var_x}, cov_x_y={cov_x_y}, theta_1={theta_1}, theta_0={theta_0}')

y_hat = theta_1 * x + theta_0
print('y_hat_1=', y_hat[0])
y_hat_new = theta_1 * 30 + theta_0
print('y_hat_new=', y_hat_new)

mse = np.average((y_hat - y) ** 2)
mae = np.average(abs(y_hat - y))
rmse = np.sqrt(mse)
print(f'mse={mse}, mae={mae}, rmse={rmse}')

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
