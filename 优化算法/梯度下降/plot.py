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
    X, Y = np.meshgrid(x, y)  # x沿着y的维度复制得到X，y沿着x的维度复制得到Y. 让X和Y重叠，重叠位置就是(x,y)坐标.
    points = np.stack([X.ravel(), Y.ravel()], axis=1)  # 网格点变为坐标点(x,y).
    points = torch.tensor(points, dtype=torch.float32)  # 转化为张量.
    Z = loss.forward(points)
    Z = Z.reshape(X.shape).numpy()  # 转换为网格数据.

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

    # 确保输入数据维度正确 (10,) 代替 (10,1)
    X_flat = X_samples.squeeze(-1)  # 形状从 (10,1) -> (10,)
    y_flat = y_samples.squeeze(-1)  # 形状从 (10,1) -> (10,)

    # 生成参数网格
    W, B = torch.meshgrid(w, b, indexing='ij')  # 形状均为 (400,400)

    # 将参数组合展开为矩阵计算格式
    theta_1 = W.reshape(-1, 1)  # (160000,1)
    theta_2 = B.reshape(-1, 1)  # (160000,1)

    # 向量化计算预测值 (160000,10)
    y_pred = theta_1 * X_flat.unsqueeze(0) + theta_2  # 广播机制

    # 计算MSE损失 (160000,)
    loss = torch.mean((y_pred - y_flat) ** 2, dim=1)

    # 恢复网格形状 (400,400)
    loss_grid = loss.reshape(W.shape)

    X, Y, Z = W.numpy(), B.numpy(), loss_grid.numpy()

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
