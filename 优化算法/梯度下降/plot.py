import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_points(points, lr, annotate=True):
    """

    :param points: 坐标点.
    :param lr: 学习率.
    :param annotate: 是否标准步骤.
    :return:
    """
    points = np.array(points)
    # 绘制参数点以及路径.
    plt.plot(points[:, 0], points[:, 1], 'o-', markersize=8, linewidth=1, label=rf'$\eta$ = {lr}')
    if annotate:
        for i, (x, y) in enumerate(points):  # 标注迭代步骤
            plt.annotate(str(i), (x, y), textcoords="offset points", xytext=(0, 10), ha='center')
    plt.legend()


def plot_contour(loss, r, n):
    # 生成等高线数据
    x = np.linspace(-r, r, n)  # 生成400个点, 包含-r和r，步长step=2r/(400-1).
    y = np.linspace(-r, r, n)

    X, Y, Z = loss.mesh(x, y)

    contour = plt.contour(X, Y, Z, levels=np.logspace(0, 3, 10), cmap='viridis')  # 绘制等高线.
    plt.clabel(contour, inline=True, fontsize=8)
    # 设置图形属性
    plt.xlim(-r, r)
    plt.ylim(-r, r)
    plt.xlabel(r'$\theta_1$')
    plt.ylabel(r'$\theta_2$')
    plt.title(r'Gradient Descent on $L(\mathbf{\theta})=\theta_1^2+2\theta_2^2$')
    plt.grid(True)


def plot_contour_with_sample(loss, X_samples, y_samples, r, n):
    """
    绘制损失函数等高线.
    :return:
    """
    w = torch.linspace(-r, r, n)
    b = torch.linspace(-r, r, n)

    X, Y, Z = loss.mesh(w, b, X_samples, y_samples)  # 计算损失.

    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
    plt.xlabel(r'$\theta_1$')
    plt.ylabel(r'$\theta_2$')
    ax1.set_zlabel(r'$L$')
    plt.title(r'Loss Function')

    ax2 = fig.add_subplot(122)
    contour = plt.contour(X, Y, Z, levels=np.logspace(0, 3, 10), cmap='viridis')  # 绘制等高线.
    plt.clabel(contour, inline=True, fontsize=8)
    # ax2.scatter(true_w, true_b, c='red', marker='x', s=100, label='True Params')

    # 设置图形属性
    plt.xlim(-r, r)
    plt.ylim(-r, r)
    plt.xlabel(r'$\theta_1$')
    plt.ylabel(r'$\theta_2$')
    plt.title(r'Gradient Descent')
    plt.grid(True)
