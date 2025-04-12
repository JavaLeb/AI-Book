import torch
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt


def generate_data():
    torch.manual_seed(42)
    n_samples = 10
    x = torch.linspace(0, 10, n_samples)  # shape: (10,).
    true_w, true_b = 2.0, 1.0
    y = true_w * x + true_b + torch.randn(x.size()) * 1.5  # shape: (10,)
    X = x.reshape(-1, 1)  # shape: (10,1)
    y = y.reshape(-1, 1)  # shape: (10,1)

    return X, y


class SquareFunction:
    """
    函数f(x,y) = c1 * x1^2 + c2 * x2^2.
    """

    def __init__(self, c: Tensor) -> None:
        """

        :param c:   shape=Tensor(n,)
        """
        self.c = c

    def forward(self, params: Tensor) -> Tensor:
        """

        :param params: shape=Tensor(n,)
        :return: 沿最后一个维度求和. 例如shape(n,)只有一个维度，求和后维度为shape=Tensor(),零维（标量）.
        """
        return torch.sum(self.c * params ** 2, dim=-1)


class MSELoss:
    def __init__(self) -> None:
        pass

    def forward(self, y_predict: Tensor, y_true: Tensor, reduction='mean') -> Tensor:
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
        self.optimizer = optimizer
        self.iteration_k = 0
        self.epoch_n = 0
        self.params = self.optimizer.params
        self.points = []
        self.loss_history = []
        self.batch_size = None

    def fit(self, X=None, y=None, epochs=10000):
        loss = self.record_init(X, y)
        for epoch in range(epochs):  # epoch相当于算法中的t.
            if self.iteration_k > 0 and abs(self.loss_history[-1] - self.loss_history[-2]) < self.optimizer.tol:
                break
            loss = self.epoch(X, y, loss, epoch)
        print(f"batch size {self.batch_size}, "
              f"learning rate {self.optimizer.lr}, Converged at epoch {self.epoch_n}, iteration {self.iteration_k}, "
              f"Optimized parameters: {np.round(self.params.detach().numpy(), 3)}, "
              f"Final loss: {self.loss_history[-1]}")

    def compute_loss(self, loss_X=None, loss_y=None):
        return self.optimizer.loss.forward(self.params)

    def record_init(self, X=None, y=None):
        self.points.append(self.params.clone().detach())  # 记录参数初始值.
        loss = self.compute_loss(loss_X=X, loss_y=y)
        self.loss_history.append(loss.item())  # 记录初始损失值.
        return loss

    def epoch(self, X, y, loss, epoch):
        self.optimizer.step(loss)  # 参数更新.
        self.points.append(self.params.clone().detach())  # 记录参数每次迭代的值.
        loss = self.compute_loss(loss_X=X, loss_y=y)
        self.loss_history.append(loss.item())  # 记录参数的损失值.
        self.iteration_k = epoch + 1  # 记录迭代步骤.
        self.epoch_n = epoch + 1

        return loss


class LinearModel(VirtualModel):

    def __init__(self, in_features: int, out_features: int, optimizer) -> None:
        super().__init__(optimizer)
        self.in_features = in_features
        self.out_features = out_features

    def _add_bias_term(self, X: Tensor) -> Tensor:
        """为输入数据添加全 1 的偏置列"""
        batch_size = X.shape[0]
        ones = torch.ones((batch_size, 1))  # 添加的常数列
        return torch.cat([X, ones], dim=1)  # 沿特征维度拼接

    def forward(self, X: Tensor) -> Tensor:
        X_augmented = self._add_bias_term(X)  # 添加偏置列
        return X_augmented @ self.params.reshape(-1, 1)  # 矩阵乘法

    def compute_loss(self, loss_X=None, loss_y=None):
        return self.optimizer.loss.forward(y_predict=self.forward(loss_X), y_true=loss_y)


class SGDLinearModel(LinearModel):
    def __init__(self, in_features: int, out_features: int, optimizer) -> None:
        self.epoch_loss = []
        self.n_samples = None
        self.n_features = None
        self.shuffled = True
        super().__init__(in_features, out_features, optimizer)

    def init_fit(self, X, batch_size, shuffled):
        self.n_samples, self.n_features = X.shape  # 获取样本数和特征数.
        self.batch_size = batch_size
        self.shuffled = shuffled

    def fit(self, X=None, y=None, epochs=10000, batch_size=10, shuffled=True):
        self.init_fit(X, batch_size, shuffled)
        super().fit(X, y, epochs)

    def epoch(self, X, y, loss, epoch):
        if self.shuffled:
            shuffled_indices = np.random.permutation(self.n_samples)  # 样本序号打乱.
        else:
            shuffled_indices = np.arange(self.n_samples)  # 固定顺序
        for start_idx in range(0, self.n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.n_samples)
            batch_indices = shuffled_indices[start_idx:end_idx]
            self.step(X, y, batch_indices)
            self.iteration_k += 1
        self.points.append(self.params.clone().detach())  # 记录参数初始值.
        loss = self.compute_loss(loss_X=X, loss_y=y)
        self.loss_history.append(loss.item())  # 记录初始损失值.
        self.epoch_n = epoch + 1

    def step(self, X, y, batch_indices):
        mini_loss = self.compute_loss(loss_X=X[batch_indices], loss_y=y[batch_indices])
        self.optimizer.step(mini_loss)  # 参数更新.


class SAGLinearModel(SGDLinearModel):
    def __init__(self, in_features: int, out_features: int, optimizer) -> None:
        super().__init__(in_features, out_features, optimizer)
        self.average_grads = None
        self.samples_grads = None

    def init_fit(self, X, batch_size, shuffled):
        super().init_fit(X, batch_size, shuffled)
        # 样本梯度，记录每个样本的梯度.
        self.samples_grads = torch.zeros(self.n_samples, self.in_features + 1)  # shape=(n_sample,feature+1)
        # 整个数据集D的梯度.
        self.average_grads = torch.mean(self.samples_grads, dim=0)  # shape=(feature+1,0)

    def step(self, X, y, batch_indices):
        mini_loss = self.optimizer.loss.forward(y_predict=self.forward(X[batch_indices]),
                                                y_true=y[batch_indices], reduction=None)
        self.optimizer.step(mini_loss=mini_loss,
                            samples_grads=self.samples_grads,
                            average_grads=self.average_grads,
                            batch_indices=batch_indices,
                            n_samples=self.n_samples)  # 参数更新.


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
