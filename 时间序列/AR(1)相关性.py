import numpy as np
from matplotlib import pyplot as plt
from pandas.plotting import lag_plot, autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf
import pandas as pd

np.random.seed(42)  # 设置种子，保证每次运行结果都一样.
num = 1001
ts = np.arange(0, num)  # 生成时间点t=0,1,...,100.
alpha_0 = 2
i = 1  # 控制折线图序号.
series_list = []
for alpha_1 in [0.9, 1, -1]:  # 不同的alpha取值.
    yt_1 = 0  # 初始化y0.
    y = []  # 存储yt的值.
    mean = 0  # 均值
    std = 1  # 标准差
    e_t = np.random.normal(mean, std, num)  # 生成num个标准正态分布随机数.
    for t in ts:
        if t == 0:
            y.append(yt_1)
            continue
        yt = alpha_0 + alpha_1 * yt_1 + e_t[t]  # 一阶线性差分方程.
        y.append(yt)
        yt_1 = yt
    series = pd.Series(data=y, index=ts)
    series_list.append(series)
    plt.subplot(3, 2, i)
    lag_plot(series, lag=1)  # 绘制(y(t),y(t+1))散点图

    plt.subplot(3, 2, i + 1)
    autocorrelation_plot(series)  # 绘制自相关图（折线图）.

    # plt.plot(ts, y)  # 绘制折线图.
    # plt.title(f'$\\alpha_1={alpha_1}$')  # 绘制折线图标题.
    # plt.xlabel(f'$t$')  # 绘制折线图x标签.
    # plt.ylabel('$y_t$')  # 绘制折线图y轴标签.
    i = i + 2
for s in series_list:
    plot_acf(s)  # 绘制自相关图.
plt.show()
