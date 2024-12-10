import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, logistic

if __name__ == '__main__':
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 显示中文
    plt.rcParams["axes.unicode_minus"] = False  # 显示负号

    plt.subplot(2, 2, 1)
    u, sigma, = 0, 1  # 标准正态分布期望和方差
    logic_sigma = sigma / 1.6  # 逻辑分布方差.
    x = np.arange(-5, 5, 0.01)

    # 标准正态分布概率密度函数f(x).
    normal_pdf = norm.pdf(x, loc=u, scale=sigma)
    normal_pdf_label = r'$f(x)=\frac{1}{\sqrt{2\pi}}e^{-\frac{x^2}{2}}$'
    plt.plot(x, normal_pdf, color='red', label=normal_pdf_label)

    # 绘制标准逻辑分布密度函数曲线s(x).
    normal_logic_pdf = logistic.pdf(x, loc=u, scale=sigma)
    normal_logic_pdf_label = r'$s(x)=\frac{e^{-x}}{(1+e^{-x})^2}$'
    plt.plot(x, normal_logic_pdf, color='green', label=normal_logic_pdf_label)

    # 绘制近似的逻辑分布密度函数s(x).
    logic_pdf = logistic.pdf(x, loc=u, scale=logic_sigma)
    logic_pdf_label = r'$s(x)=\frac{1.6e^{-1.6x}}{(1+e^{-1.6x})^2}$'
    plt.plot(x, logic_pdf, color='blue', label=logic_pdf_label)

    plt.legend()  # 显示图示
    plt.xlabel('x')
    plt.ylabel('f(x)/s(x)')
    plt.title('标准正态分布与逻辑分布概率密度函数对比图', fontproperties="SimHei")

    # 绘制正态分布和逻辑分布累计分布函数曲线.
    plt.subplot(2, 2, 2)
    # 绘制标准正态分布累计分布函数F(x).
    normal_cdf = norm.cdf(x, loc=u, scale=sigma)
    normal_cdf_label = r'$F(x)=\int_{-\infty }^{x}\frac{1}{\sqrt{2\pi}}e^{-\frac{t^2}{2}}dt$'
    plt.plot(x, normal_cdf, color='red', label=normal_cdf_label)

    # 绘制标准逻辑分布累计分布函数S(x).
    normal_logic_cdf = logistic.cdf(x, loc=u, scale=sigma)
    normal_logic_cdf_label = r'$S(x)=\frac{1}{1+e^{-x}}$'
    plt.plot(x, normal_logic_cdf, color='green', label=normal_logic_cdf_label)

    # 绘制逻辑分布累计分布函数S(x).
    logic_cdf = logistic.cdf(x, loc=u, scale=logic_sigma)
    logic_cdf_label = r'$S(x)=\frac{1}{1+e^{-1.6x}}$'
    plt.plot(x, logic_cdf, color='blue', label=logic_cdf_label)

    plt.legend()  # 显示图示
    plt.xlabel('x')
    plt.ylabel('F(x)/S(x)')
    plt.title('标准正态分布与逻辑分布累计分布函数对比图', fontproperties="SimHei")
    plt.show()
