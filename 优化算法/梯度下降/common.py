import torch
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt


def generate_data():
    """
    生成一元线性回归样本数据.
    :return: 数据.
    """
    torch.manual_seed(42)  # 设置种子，保证每次运行生成的数据一样.
    n_samples = 10  # 样本数.
    x = torch.linspace(0, 10, n_samples)  # 生成特征数据. shape:(10,).
    true_w, true_b = 2.0, 1.0  # 参数的理论值.
    y = true_w * x + true_b + torch.randn(x.size()) * 1.5  # 生成标签. shape: (10,)
    X, y = x.reshape(-1, 1), y.reshape(-1, 1)  # 变换数据的维度. shape: (10,1)

    return X, y


class SquareFunction:
    """
    函数f(x1,x2,...,xn) = c1 * x1^2 + c2 * x2^2 + ... + cn * xn^2.
    """

    def __init__(self, c: Tensor) -> None:
        """

        :param c:   系数, shape=Tensor(n,).
        """
        self.c = c

    def forward(self, params: Tensor) -> Tensor:
        """
        计算函数值.
        :param params: 函数自变量, shape=Tensor(n,).
        :return: 沿最后一个维度求和. 例如shape(n,)只有一个维度，求和后维度为shape=Tensor(),零维（标量）.
        """
        return torch.sum(self.c * params ** 2, dim=-1)


class MSELoss:
    """
    均方误差损失函数.
    """

    def __init__(self) -> None:
        pass

    def forward(self, y_predict: Tensor, y_true: Tensor, reduction='mean') -> Tensor:
        """

        :param y_predict: 预测标签值值.
        :param y_true: 真实标签值.
        :param reduction: 如何计算损失，mean: 平均，sum: 和，其它值: 什么也不做.
        :return:
        """
        if reduction == 'mean':
            return torch.mean((y_predict - y_true) ** 2)
        elif reduction == 'sum':
            return torch.sum((y_predict - y_true) ** 2)
        else:
            return (y_predict - y_true) ** 2


class VirtualModel:
    """
    虚拟模型.
    """

    def __init__(self, optimizer) -> None:
        """

        :param optimizer: 优化器.
        """
        self.loss = None  # 损失值.
        self.optimizer = optimizer  # 优化器.
        self.iteration_k = 0  # 最终的迭代k.
        self.epoch_n = 0  # 轮数，遍历一遍数据集，算一轮.
        self.params = self.optimizer.params  # 优化器参数.
        self.points = []  # 参数点.
        self.loss_history = []  # 历史损失值.
        self.batch_size = None  # 批大小.

    def fit(self, X=None, y=None, epochs=10000):
        """

        :param X: 特征数据.
        :param y: 标签数据.
        :param epochs: 训练轮数.
        :return:
        """
        self.record()  # 记录参数点和损失值.
        for epoch in range(epochs):  # 循环训练.
            # 训练退出条件：当前损失值与上一次损失值误差小于给定阈值则退出.
            if self.iteration_k > 0 and abs(self.loss_history[-1] - self.loss_history[-2]) < self.optimizer.tol:
                break
            self.epoch(X, y, epoch)
        print(f"batch size {self.batch_size}, "
              f"learning rate {self.optimizer.lr}, Converged at epoch {self.epoch_n}, iteration {self.iteration_k}, "
              f"Optimized parameters: {np.round(self.params.detach().numpy(), 3)}, "
              f"Final loss: {self.loss_history[-1]}")

    def record(self):
        self.points.append(self.params.clone().detach())  # 记录参数初始值.
        self.loss = self.optimizer.loss.forward(self.params)  # 计算初始参数的损失值.
        self.loss_history.append(self.loss.item())  # 记录初始损失值.

    def epoch(self, X, y, epoch):
        """
        一轮.
        :param X: 特征数据.
        :param y: 标签数据.
        :param epoch: 当前轮.
        :return:
        """
        self.optimizer.step(model=self)  # 参数优化.
        self.record()  # 记录参数点和损失值.
        self.iteration_k = epoch + 1
        self.epoch_n = epoch + 1


class LinearModel(VirtualModel):
    def __init__(self, in_features: int, out_features: int, optimizer) -> None:
        """
        线性模型初始化方法.
        :param in_features: 输入特征数.
        :param out_features: 输出标签数.
        :param optimizer: 优化器.
        """
        super().__init__(optimizer)
        self.y = None  # 标签数据.
        self.X = None  # 特征数据.
        self.batch_indices = None  # 每次随机选取的样本索引.
        self.in_features = in_features  # 输入特征数.
        self.out_features = out_features  # 输出特征数.
        self.epoch_loss = []  # 每轮的损失.
        self.n_samples = None  # 样本数.
        self.n_features = None  # 特征数.
        self.shuffled = True  # 是否打乱数据.
        self.average_grads = None  # 平均梯度.
        self.samples_grads = None  # 样本梯度.

    def forward(self, X: Tensor) -> Tensor:
        """
        计算预测标签值.
        :param X: 特征值.
        :return: 预测标签值.
        """

        ones = torch.ones((X.shape[0], 1))  # 添加的常数列
        X_augmented = torch.cat([X, ones], dim=1)  # 沿特征维度拼接

        return X_augmented @ self.params.reshape(-1, 1)  # 矩阵乘法

    def fit(self, X=None, y=None, epochs=10000, batch_size=None, shuffled=True):
        self.X, self.y = X, y
        self.n_samples, self.n_features = X.shape  # 获取样本数和特征数.
        if batch_size is not None:
            self.batch_size = batch_size
        else:
            self.batch_size = self.n_samples
        self.shuffled = shuffled
        # 样本梯度，记录每个样本的梯度.
        self.samples_grads = torch.zeros(self.n_samples, self.in_features + 1)  # shape=(n_sample,feature+1)
        # 整个数据集D的梯度.
        self.average_grads = torch.mean(self.samples_grads, dim=0)  # shape=(feature+1,0)

        super().fit(X, y, epochs)

    def epoch(self, X, y, epoch):
        if self.shuffled:
            shuffled_indices = np.random.permutation(self.n_samples)  # 样本序号打乱.
        else:
            shuffled_indices = np.arange(self.n_samples)  # 固定顺序
        for start_idx in range(0, self.n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.n_samples)
            self.batch_indices = shuffled_indices[start_idx:end_idx]
            self.optimizer.step(model=self)  # 参数更新.
            self.iteration_k += 1
        self.record()
        self.epoch_n = epoch + 1

    def record(self):
        self.points.append(self.params.clone().detach())  # 记录参数初始值.
        self.loss = self.optimizer.loss.forward(y_predict=self.forward(self.X), y_true=self.y)
        self.loss_history.append(self.loss.item())  # 记录初始损失值.


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
