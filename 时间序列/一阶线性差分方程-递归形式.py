import numpy as np
from matplotlib import pyplot as plt

num = 101
ts = np.arange(0, num)  # 生成时间点t=0,1,...,100.
i = 1  # 控制折线图序号.
for alpha_1 in [0.01, 1.01, 1, -1]:  # 不同的alpha取值.
    yt_1 = 0  # 初始化y0.
    y = []  # 存储yt的值.
    mean = 0  # 均值
    std = 1  # 标准差
    np.random.seed(42)  # 设置种子，保证每次运行结果都一样.
    w_t = np.random.normal(mean, std, num)  # 生成num个标准正态分布随机数.
    for t in ts:
        if t == 0:
            y.append(yt_1)
            continue
        yt = alpha_1 * yt_1 + w_t[t]  # 一阶线性差分方程.
        y.append(yt)
        yt_1 = yt
    plt.subplot(2, 2, i)
    plt.plot(ts, y)  # 绘制折线图.
    plt.title(f'$\\alpha_1={alpha_1}$')  # 绘制折线图标题.
    plt.xlabel(f'$t$')  # 绘制折线图x标签.
    plt.ylabel('$y_t$')  # 绘制折线图y轴标签.
    i = i + 1
plt.show()