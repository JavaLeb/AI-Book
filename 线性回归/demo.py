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

# 绘制散点图
plt.rcParams['font.sans-serif'] = ['SimHei']  # 避免散点图显示中文乱码.
plt.scatter(x, y)


# 添加标题和坐标轴标签
plt.title("产品个数-生产成本散点图")
plt.xlabel("产品个数")
plt.ylabel("生产成本")

# 显示图形
plt.show()
