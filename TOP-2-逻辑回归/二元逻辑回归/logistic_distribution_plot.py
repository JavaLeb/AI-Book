import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, logistic

if __name__ == '__main__':
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 显示中文
    plt.rcParams["axes.unicode_minus"] = False  # 显示负号

    plt.subplot(2, 2, 1)
    x = np.arange(-10, 10, 0.01)
    plt.xlim(-10, 10)
    plt.ylim(0, 0.3)
    # 绘制标准逻辑分布密度函数曲线s(x).
    u, sigma, = 0, 1  # 标准正态分布期望和方差
    normal_logic_pdf = logistic.pdf(x, loc=u, scale=sigma)
    label = rf'$u={u},\sigma={sigma}$'
    plt.plot(x, normal_logic_pdf, label=label,color = 'red')
    plt.plot([u, u], [0, 0.3], linestyle='--', color='gray')

    u, sigma, = 0, 2  # 标准正态分布期望和方差
    normal_logic_pdf = logistic.pdf(x, loc=u, scale=sigma)
    label = rf'$u={u},\sigma={sigma}$'
    plt.plot(x, normal_logic_pdf, label=label,color = 'green')

    u, sigma, = 1, 1  # 标准正态分布期望和方差
    normal_logic_pdf = logistic.pdf(x, loc=u, scale=sigma)
    label = rf'$u={u},\sigma={sigma}$'
    plt.plot(x, normal_logic_pdf, label=label,color = 'blue')
    plt.plot([u, u], [0, 0.3], linestyle='--', color='gray')



    plt.legend()  # 显示图示
    plt.xlabel('x')
    plt.ylabel('s(x)')
    plt.title('逻辑分布概率密度函数s(x)', fontproperties="SimHei")

    # 绘制正态分布和逻辑分布累计分布函数曲线.
    plt.subplot(2, 2, 2)
    plt.xlim(-10, 10)
    plt.ylim(0, 1)
    # 绘制标准逻辑分布累计分布函数S(x).
    u, sigma, = 0, 1
    normal_logic_cdf = logistic.cdf(x, loc=u, scale=sigma)
    label = rf'$u={u},\sigma={sigma}$'
    plt.plot(x, normal_logic_cdf, label=label,color = 'red')
    plt.plot([u, u], [0, 1], linestyle='--', color='gray')

    u, sigma, = 0, 2
    normal_logic_cdf = logistic.cdf(x, loc=u, scale=sigma)
    label = rf'$u={u},\sigma={sigma}$'
    plt.plot(x, normal_logic_cdf, label=label,color = 'green')
    label = rf'对称中心({u},0.5)'
    plt.scatter(u, 0.5, label=label)

    u, sigma, = 1, 1
    normal_logic_cdf = logistic.cdf(x, loc=u, scale=sigma)
    label = rf'$u={u},\sigma={sigma}$'
    plt.plot(x, normal_logic_cdf, label=label,color = 'blue')
    label = rf'对称中心({u},0.5)'
    plt.scatter(u, 0.5, label=label)
    plt.plot([-10, 10], [0.5, 0.5], linestyle='--', color='gray')
    plt.plot([u, u], [0, 1], linestyle='--', color='gray')

    plt.legend()  # 显示图示
    plt.xlabel('x')
    plt.ylabel('S(x)')
    plt.title('逻辑分布累计分布函数S(x)', fontproperties="SimHei")
    plt.show()
