import numpy as np
from matplotlib import pyplot as plt

num = 101
ts = np.arange(0, num)
i = 1
for alpha_1 in [0.01, 1.01, 1, -1]:
    plt.subplot(2, 2, i)
    yt_1 = 0
    y = []
    mean = 0  # 均值
    std = 1  # 标准差
    np.random.seed(42)
    w_t = np.random.normal(mean, std, num)
    for t in ts:
        if t == 0:
            y.append(yt_1)
            continue
        yt = alpha_1 * yt_1 + w_t[t]
        y.append(yt)
        yt_1 = yt
    plt.plot(ts, y)
    plt.title(f'$\\alpha_1={alpha_1}$')
    plt.xlabel(f'$t$')
    plt.ylabel('$y_t$')
    i = i + 1
plt.show()

x = np.arange(0, 21)
alphas = [0.9, -0.9, 1, -1, 1.1, -1.1]

for i in range(0, len(alphas)):
    if i % 4 == 0:
        plt.figure()
    y = np.power(alphas[i], x)
    plt.subplot(2, 2, i % 4 + 1)
    plt.bar(x, y, width=0.5)
    plt.title(f'$\\alpha_1={alphas[i]}$')
    plt.xlabel(f'$h$')
    plt.ylabel('${\\alpha_1}^h$')
plt.show()
