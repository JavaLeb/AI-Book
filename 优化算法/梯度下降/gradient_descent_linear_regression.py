import numpy as np
import matplotlib.pyplot as plt
import torch
from gradient_descent import gradient_descent


def plot_points(points, lr):
    points = np.array(points)
    # 绘制参数点以及路径.
    plt.plot(points[:, 0], points[:, 1], 'o-', markersize=5, linewidth=1, label=rf'$\eta$ = {lr}')
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
    plt.title(r'Loss Function')

    ax2=fig.add_subplot(122)
    contour = plt.contour(X, Y, Z, levels=np.logspace(0, 3, 10), cmap='viridis')  # 绘制等高线.
    plt.clabel(contour, inline=True, fontsize=8)
    ax2.scatter(true_w, true_b, c='red', marker='x', s=100, label='True Params')

    # 设置图形属性
    r = 11
    plt.xlim(-r, r)
    plt.ylim(-r, r)
    plt.xlabel(r'$\theta_1$')
    plt.ylabel(r'$\theta_2$')
    plt.title(r'Gradient Descent')
    plt.grid(True)


torch.manual_seed(42)
x = torch.linspace(0, 10, 10)  # shape: (10,).
true_w, true_b = 2.0, 1.0
y = true_w * x + true_b + torch.randn(x.size()) * 1.5  # shape: (10,)



def loss_func(params):
    """
    优化后的损失函数，支持单点和网格参数输入
    参数：
    params : Tensor of shape (2,) or (N, 2) 或 (H, W, 2)
    返回：
    loss : 标量或 (H, W) 的损失矩阵
    """
    # 统一输入维度为 (..., 2) (优化点2：灵活维度处理)
    original_shape = params.shape
    params = params.reshape(-1, 2)  # 展平为 (N, 2)

    theta_1 = params[:, 0]  # shape: (N,)
    theta_2 = params[:, 1]  # shape: (N,)

    # 利用广播机制批量计算 (优化点3：避免冗余reshape)
    # x.shape=(10,) => (1, 10), w/b.shape=(N,1) => 最终shape=(N,10)
    y_pred = theta_1[:, None] * x[None, :] + theta_2[:, None]

    # 计算均方误差 (优化点4：高效矩阵运算)
    loss = torch.mean((y_pred - y[None, :]) ** 2, dim=1)

    # 恢复原始形状
    return loss.reshape(original_shape[:-1])


def mesh_loss_func():
    """
    生成参数网格和对应的损失矩阵 (优化点5：独立网格生成函数)
    参数：
    w_range : 权重参数范围
    b_range : 偏置参数范围
    steps   : 每个方向的采样点数
    """
    # 生成参数网格 (优化点6：使用更合理的参数范围)
    r, n = 11, 400
    theta_1 = torch.linspace(-r, r, n)
    theta_2 = torch.linspace(-r, r, n)
    W, B = torch.meshgrid(theta_1, theta_2, indexing='ij')  # shape: (steps, steps)

    # 合并网格点 (优化点7：高效内存布局)
    grid = torch.stack([W, B], dim=-1)  # shape: (steps, steps, 2)

    # 计算损失
    with torch.no_grad():
        loss = loss_func(grid)

    return W.numpy(), B.numpy(), loss.numpy()


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
