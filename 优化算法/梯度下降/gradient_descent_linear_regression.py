import numpy as np
import matplotlib.pyplot as plt
import torch
from gradient_descent import gradient_descent


def plot_points(points, lr):
    points = np.array(points)
    # 绘制参数点以及路径.
    plt.plot(points[:, 0], points[:, 1], 'o-', markersize=4, linewidth=1, label=rf'$\eta$ = {lr}')
    plt.legend()


def plot():
    """
    绘制损失函数等高线.
    :return:
    """
    X, Y, Z = mesh_loss_func()  # 计算损失.
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
    plt.xlabel(r'$\theta_1$')
    plt.ylabel(r'$\theta_2$')
    ax1.set_zlabel(r'$L$')
    plt.title(r'Gradient Descent')
    fig.add_subplot(122)
    contour = plt.contour(X, Y, Z, levels=np.logspace(0, 3, 10), cmap='viridis')  # 绘制等高线.
    plt.clabel(contour, inline=True, fontsize=8)
    # 设置图形属性
    r = 11
    plt.xlim(-r, r)
    plt.ylim(-r, r)
    plt.xlabel(r'$\theta_1$')
    plt.ylabel(r'$\theta_2$')
    plt.title(r'Gradient Descent')
    plt.grid(True)


# 生成一元线性回归样本数据.
torch.manual_seed(42)
x = torch.linspace(0, 10, 10)  # shape: (10,).
true_w, true_b = 2.0, 1.0
y = true_w * x + true_b + torch.randn(x.size()) * 1.5  # shape: (10,)


def loss_func(params):
    """
    计算参数网格上的损失值
    参数：
    w_grid : 权重参数网格 (100, 100)
    b_grid : 偏置参数网格 (100, 100)
    返回：
    loss : 损失值网格 (100, 100)
    """
    # 维度调整（关键修正步骤）
    w_grid = params[0]  # theta1
    b_grid = params[1]  # theta2

    x_3d = x.reshape(1, 1, -1)  # shape: (1, 1, 10)
    w_3d = w_grid.unsqueeze(-1)  # shape: (100, 100, 1)
    b_3d = b_grid.unsqueeze(-1)  # shape: (100, 100, 1)

    # 广播计算预测值 (100, 100, 10)
    y_pred = w_3d * x_3d + b_3d

    # 计算损失 (100, 100)
    loss = torch.mean((y_pred - y) ** 2, dim=2)
    return loss


def mesh_loss_func():
    """
    计算参数网格上的损失值
    参数：
    w_grid : 权重参数网格 (100, 100)
    b_grid : 偏置参数网格 (100, 100)
    返回：
    loss : 损失值网格 (100, 100)
    """
    # 维度调整（关键修正步骤）
    # 生成参数网格 (修正索引方式)
    r = 10
    x = torch.linspace(-r, r, 400)  # shape: (100,)
    y = torch.linspace(-r, r, 400)  # shape: (100,)
    X, Y = torch.meshgrid(x, y, indexing='ij')  # shape: (100, 100)

    x_3d = x.reshape(1, 1, -1)  # shape: (1, 1, 10)
    w_3d = X.unsqueeze(-1)  # shape: (100, 100, 1)
    b_3d = Y.unsqueeze(-1)  # shape: (100, 100, 1)

    # 广播计算预测值 (100, 100, 10)
    y_pred = w_3d * x_3d + b_3d

    # 计算损失 (100, 100)
    loss = torch.mean((y_pred - y) ** 2, dim=2)
    return X, Y, loss


if __name__ == "__main__":
    initial_params = torch.tensor([10.0, 10.0], requires_grad=False)  # 初始化参数
    lr = 0.01
    # 运行梯度下降.
    _, _, points = gradient_descent(func=loss_func, init_params=initial_params,
                                    lr=lr,
                                    max_iters=10000)
    plot()  # 绘制等高线.
    plot_points(points, lr)  # 绘制参数迭代路径.
    plt.show()
