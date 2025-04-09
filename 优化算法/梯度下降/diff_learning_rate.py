import numpy as np
import matplotlib.pyplot as plt
import torch
from gradient_descent import gradient_descent


def plot_points(points, lr):
    points = np.array(points)
    # 绘制参数点以及路径.
    plt.plot(points[:, 0], points[:, 1], 'o-', markersize=8, linewidth=1, label=rf'$\eta$ = {lr}')

    for i, (x, y) in enumerate(points):  # 标注迭代步骤
        plt.annotate(str(i), (x, y), textcoords="offset points", xytext=(0, 10), ha='center')
    plt.legend()


def plot_contour():
    """
    绘制损失函数等高线.
    :return:
    """
    X, Y, Z = mesh_loss_func()  # 计算损失.
    contour = plt.contour(X, Y, Z, levels=np.logspace(0, 3, 10), cmap='viridis')  # 绘制等高线.
    plt.clabel(contour, inline=True, fontsize=8)
    # 设置图形属性
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    plt.xlabel(r'$\theta_1$')
    plt.ylabel(r'$\theta_2$')
    plt.title(r'Gradient Descent on $L(\mathbf{\theta})=\theta_1^2+2\theta_2^2$')
    plt.grid(True)


def mesh_loss_func():
    # 生成等高线数据
    r, n = 15, 400
    x = np.linspace(-r, r, n)  # 生成400个点, 包含-r和r，步长step=2r/(400-1).
    y = np.linspace(-r, r, n)
    X, Y = np.meshgrid(x, y)  # x沿着y的维度复制得到X，y沿着x的维度复制得到Y. 让X和Y重叠，重叠位置就是(x,y)坐标.

    points = np.stack([X.ravel(), Y.ravel()], axis=1)  # 网格点变为坐标点(x,y).
    points = torch.tensor(points, dtype=torch.float32)  # 转化为张量.

    Z = loss_func(points)
    Z = Z.reshape(X.shape).numpy()  # 转换为网格数据.

    return X, Y, Z


def loss_func(params):
    """
    定义损失函数 f(x,y) = x1^2 + 2 * x2^2.
    :param params: 函数自变量张量，例如[x1,x2].
    :return: 函数因变量.
    """
    c = torch.tensor([1, 2])

    return torch.sum(c * params ** 2, dim=-1)


if __name__ == "__main__":
    initial_params = torch.tensor([[10.0, 10.0], [-10.0, 10.0], [-10.0, -10.0]], requires_grad=False)  # 初始化参数
    learning_rates = [0.5, 0.45, 0.1]
    # 运行梯度下降.
    plt.figure(figsize=(10, 8))
    for i in range(len(learning_rates)):
        lr = learning_rates[i]
        _, _, points = gradient_descent(func=loss_func,
                                        init_params=initial_params[i],
                                        lr=lr,
                                        max_iters=10000)
        plot_points(points, lr)  # 绘制参数迭代路径.
    plot_contour()  # 绘制等高线.
    plt.show()
