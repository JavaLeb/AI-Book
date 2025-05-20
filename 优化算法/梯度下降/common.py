import torch
import torch_optimizer
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
from optimizer import SystemOptimizer, NAGOptimizer


class SquareFunction:
    """
    多元函数.
    f(x1,x2,...,xn) = c1 * x1^2 + c2 * x2^2 + ... + cn * xn^2.
    """

    def __init__(self, coefficient: Tensor) -> None:
        """
        :param coefficient:   系数, shape=Tensor(n,).
        """
        self.coefficient = coefficient

    def forward(self, params: Tensor) -> Tensor:
        """
        计算函数值.
        :param params: 函数自变量, shape=Tensor(n,).
        :return: 函数值, dim=-1表示沿着最后一个维度求和，是为了支持网格数据计算.
        """
        return torch.sum(self.coefficient * params ** 2)

    def forward_mesh(self, params: Tensor) -> Tensor:
        return torch.sum(self.coefficient * params ** 2, dim=-1)


class MSELoss:
    """
    均方误差损失函数.
    """

    def __init__(self) -> None:
        self.loss = None
        self.grad = None

    def forward(self, y_predict: Tensor, y_true: Tensor, reduction='mean') -> Tensor:
        """

        :param y_predict: 预测值.
        :param y_true: 真实值.
        :param reduction: 如何计算损失，mean: 平均，sum: 和，其它值: 什么也不做.
        :return:
        """
        sub = (y_predict - y_true) ** 2 / 2.0
        if reduction == 'mean':
            self.loss = torch.mean(sub)
        elif reduction == 'sum':
            self.loss = torch.sum(sub)
        else:
            self.loss = sub

        return self.loss

    def forward_mesh(self, y_predict: Tensor, y_true: Tensor, reduction='mean') -> Tensor:
        return self.forward(y_predict=y_predict, y_true=y_true, reduction=reduction)

    def backward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        计算损失对预测值的梯度 dL/dy_pred.
        :param y_pred: 预测值.
        :param y_true: 真实值.
        :return: 梯度.
        """
        self.grad = 2 * (y_pred - y_true)

        return self.grad


class VirtualModel:
    """
    虚拟模型.
    封装训练逻辑.
    """

    def __init__(self, criterion, optimizer) -> None:
        """
        初始化方法.
        :param criterion: 损失函数.
        :param optimizer: 优化器.
        """
        self.loss = None  # 损失值.
        self.optimizer = optimizer  # 优化器.
        self.criterion = criterion  # 损失函数.
        self.iteration_t = 0  # 迭代次数.
        self.epoch_t = 0  # 轮数，遍历一遍数据集，算一轮.
        self.params = self.optimizer.params  # 模型参数.
        self.points = []  # 参数点.
        self.loss_history = []  # 历史损失值.
        self.batch_size = None  # 批量大小.

    def fit(self, epochs=2000) -> None:
        """
        训练逻辑.
        :param epochs: 训练轮数.
        :return:
        """
        from torch.optim import LBFGS
        is_closure = isinstance(self.optimizer, SystemOptimizer) and isinstance(self.optimizer.optimizer, LBFGS)
        if is_closure:  # 闭包类型的优化算法.
            op = self.optimizer.optimizer

            def closure():  # 闭包.
                op.zero_grad()  # 梯度清零.
                loss = self.criterion.forward(self.params)  # 计算损失.
                loss.backward()  # 自动计算梯度.
                self.optimizer.optimized_params = self.params  # 最优参数.
                self.record()  # 记录信息.
                self.iteration_t += 1
                self.epoch_t += 1

                return loss

            op.step(closure)  # 一次执行整个闭包.
        else:  # 非闭包类型的优化算法.
            self.record()  # 记录初始参数和损失值.
            for epoch in range(epochs):  # 循环训练.
                self.train_epoch(epoch)  # 单轮训练.
                if len(self.loss_history) > 1 and self.quit_condition():  # 检查退出机制.
                    break
        print(f"optimizer {self.optimizer.name}, "
              f"batch_size {self.batch_size}, "
              f"learning rate {self.optimizer.lr}, Converged at epoch {self.epoch_t}, iteration {self.iteration_t}, "
              f"Optimized parameters: {np.round(self.optimizer.optimized_params.detach().numpy(), 3)}, "
              f"Final loss: {np.round(self.loss_history[-1], 3)}")

    def record(self) -> None:
        """
        记录参数值和损失值.
        :return:
        """
        from optimizer import ShampooOptimizer
        if isinstance(self.optimizer, ShampooOptimizer) or (
                isinstance(self.optimizer, SystemOptimizer) and isinstance(self.optimizer.optimizer,
                                                                           torch_optimizer.Shampoo)):
            self.points.append(self.params.clone().detach()[0])  # 记录参数值.
        else:
            self.points.append(self.params.clone().detach())  # 记录参数值.
        self.loss = self.criterion.forward(self.params)  # 计算损失值.
        self.loss_history.append(self.loss.item())  # 记录损失值.

    def train_epoch(self, epoch) -> None:
        """
        训练一轮.
        :param epoch: 当前轮次.
        :return:
        """

        if isinstance(self.optimizer, NAGOptimizer):
            def closure():  # 闭包.
                self.optimizer.zero_grad()
                loss = self.criterion.forward(self.params)  # 计算损失.
                loss.backward()

                return loss

            self.optimizer.step(closure)
        else:
            self.optimizer.optimize(self.loss)  # 优化一步.

        self.record()  # 记录优化后的参数和损失值.
        self.iteration_t = epoch + 1
        self.epoch_t = epoch + 1

    def quit_condition(self) -> bool:
        """
        退出条件.
        :return:
        """

        if self.params.grad is not None:  # 梯度接近于0.
            grad_stop = self.params.grad.abs().max() < self.optimizer.tol_grad
            if grad_stop:
                return True
        if len(self.points) > 1:  # 参数更新接近于0.
            data_change_stop = torch.abs(self.points[-1] - self.points[-2]).max() < self.optimizer.tol
            if data_change_stop:
                return True
        if len(self.loss_history) > 1:  # 损失变化接近于0.
            loss_change = self.loss_history[-1] - self.loss_history[-2]
            loss_change_stop = abs(loss_change) < self.optimizer.tol
            return loss_change_stop

        return False


class LinearRegression(VirtualModel):
    """
    线性回归模型.
    """

    def __init__(self, in_features: int, out_features: int, criterion, optimizer) -> None:
        """
        初始化.
        :param in_features: 输入特征数.
        :param out_features: 输出标签数.
        """
        super().__init__(criterion, optimizer)
        self.X = None  # 特征数据.
        self.y = None  # 标签数据.
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
        计算预测值.
        :param X: 特征值.
        :return: 预测值.
        """

        ones = torch.ones((X.shape[0], 1))  # 添加的常数列[1].
        X_augmented = torch.cat([X, ones], dim=1)  # 在最后一列添加常数项得到[X,1].

        return X_augmented @ self.params.reshape(-1, 1)  # 线性回归计算预测值y_hat=X*theta.

    def fit(self, X=None, y=None, epochs=2000, batch_size=None, shuffled=True) -> None:
        """
        拟合方法.
        :param X: 样本特征.
        :param y: 样本标签.
        :param epochs: 训练轮数.
        :param batch_size: 批大小.
        :param shuffled: 是否打乱.
        :return:
        """
        self.X, self.y = X, y
        self.n_samples, self.n_features = X.shape  # 获取样本数和特征数.
        self.batch_size = self.n_samples if batch_size is None else batch_size
        self.shuffled = shuffled
        # 样本梯度，记录每个样本的梯度, shape=(n_sample,feature+1).
        self.samples_grads = torch.zeros(self.n_samples, self.in_features + 1)
        # 按照样本维度计算整个数据集D的平均梯度, shape=(feature+1,0).
        self.average_grads = torch.mean(self.samples_grads, dim=0)
        super().fit(epochs)

    def train_epoch(self, epoch):
        rng = np.random.RandomState(seed=42)  # 保证每次随机结果一样.
        shuffled_indices = rng.permutation(self.n_samples) if self.shuffled else np.arange(self.n_samples)  # 打乱样本.
        # 整个循环表示一轮，每次循环表示一个批次.
        for start_idx in range(0, self.n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.n_samples)
            self.batch_indices = shuffled_indices[start_idx:end_idx]  # 获取一个批次样本索引.
            self.optimizer.optimize_model(self)
            self.iteration_t += 1
        self.record()  # 每训练一轮记录信息.
        self.epoch_t = epoch + 1

    def record(self):
        self.points.append(self.params.clone().detach())  # 记录参数值.
        self.loss = self.criterion.forward(y_predict=self.forward(self.X), y_true=self.y)  # 整个数据集损失.
        self.loss_history.append(self.loss.item())  # 记录损失值.


def plot_points(points, optimizer, annotate=True):
    """

    :param optimizer:
    :param points: 坐标点.
    :param annotate: 是否标准步骤.
    :return:
    """
    points = np.array(points)
    # 绘制参数点以及路径.
    plt.plot(points[:, 0], points[:, 1], 'o-', markersize=6, linewidth=0.8,
             label=rf'{optimizer.name}($\eta$ = {optimizer.lr})')
    if annotate:
        for i, (x, y) in enumerate(points):  # 标注迭代步骤
            plt.annotate(str(i), (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
    plt.legend()


def plot_contour(loss, r=15, n=400):
    # 生成等高线数据
    x = np.linspace(-r, r, n)  # 生成400个点, 包含-r和r，步长step=2r/(400-1).
    y = np.linspace(-r, r, n)
    X, Y = np.meshgrid(x, y)  # x沿着y的维度复制得到X，y沿着x的维度复制得到Y. 让X和Y重叠，重叠位置就是(x,y)坐标.
    points = np.stack([X.ravel(), Y.ravel()], axis=1)  # 网格点变为坐标点(x,y).
    points = torch.tensor(points, dtype=torch.float32)  # 转化为张量.
    Z = loss.forward_mesh(points)
    Z = Z.reshape(X.shape).numpy()  # 转换为网格数据.
    plt.figure(figsize=(10, 8))
    contour = plt.contour(X, Y, Z, levels=np.logspace(0, 3, 10), cmap='viridis')  # 绘制等高线.
    plt.clabel(contour, inline=True, fontsize=8)
    # 设置图形属性
    plt.xlim(-r, r)
    plt.ylim(-r, r)
    plt.xlabel(r'$\theta_1$')
    plt.ylabel(r'$\theta_2$')
    plt.title(r'Iterative path on $L(\mathbf{\theta})=\theta_1^2+2\theta_2^2$')
    plt.grid(True)


def plot_contour_with_sample(X_samples, y_samples, r=11, n=400):
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

    fig.add_subplot(122)
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


def plot_curve(model):
    loss_history = model.loss_history
    optimizer = model.optimizer
    plt.plot(loss_history, label=rf'{optimizer.name}($\eta$ = {optimizer.lr})')  # 学习曲线.
    # 添加最后一个点的标记和坐标
    if len(loss_history) > 0:
        x_last = len(loss_history) - 1  # x坐标为最后一个epoch的索引
        y_last = loss_history[-1]  # y坐标为最后一个loss值
        # 用红色圆点突出显示最后一个点
        plt.scatter(x_last, y_last, zorder=5, label=rf'final point {optimizer.name}($\eta$ = {optimizer.lr})')
        # 在点右侧标注坐标（保留两位小数）
        plt.annotate(
            text=f'({x_last}, {y_last:.2f})',
            xy=(x_last, y_last),
            xytext=(-20, 5),  # 向左偏移20点，向下偏移5点
            textcoords='offset points',
            fontsize=9,
            ha='right'  # 水平对齐方式
        )
    plt.xlabel("Epochs", fontsize=10)
    plt.ylabel("Loss", fontsize=10)
    plt.title("Training Loss Curve", fontsize=10)
    plt.legend()


class VMOptimizerComparator:
    """
    虚拟模型优化比较器.
    """

    def __init__(self, optimizers: list):
        self.optimizers = optimizers
        self.criterion = SquareFunction(torch.tensor([1.0, 2.0]))
        self.annotate = True

    def compare(self):
        self.plot_contour()  # 绘制等高线.
        models = []
        for optimizer in self.optimizers:
            model = self.init_model(optimizer)
            models.append(model)
            plot_points(model.points, model.optimizer, annotate=self.annotate)  # 绘制参数迭代路径.
        plt.figure(figsize=(10, 8))
        for model in models:
            plot_curve(model)
        plt.show()

    def plot_contour(self):
        plot_contour(self.criterion, r=15, n=400)  # 绘制等高线.

    def init_model(self, optimizer):
        model = VirtualModel(criterion=self.criterion, optimizer=optimizer)
        model.fit()
        return model


class LROptimizerComparator(VMOptimizerComparator):
    """
    线性回归模型优化比较器.
    """

    def __init__(self, optimizers: list):
        super().__init__(optimizers)
        self.X, self.y = generate_data()
        self.criterion = MSELoss()
        self.annotate = False

    def plot_contour(self):
        plot_contour_with_sample(self.X, self.y, r=11, n=400)  # 绘制等高线.

    def init_model(self, optimizer):
        model = LinearRegression(in_features=1, out_features=1, criterion=self.criterion, optimizer=optimizer)
        model.fit(self.X, self.y, epochs=2000, batch_size=5, shuffled=True)
        return model


def generate_data():
    """
    生成一元线性回归样本数据.
    :return: 数据.
    """
    torch.manual_seed(42)  # 设置种子，保证每次运行生成的数据一样.
    n_samples = 10  # 样本数.
    x = torch.linspace(0, 10, n_samples)  # 生成特征数据. shape:(10,).
    true_w, true_b = 2.0, 1.0  # 参数的理论值, 对应线性y=2x+1.
    y = true_w * x + true_b + torch.randn(x.size()) * 1.5  # 添加正态分布干扰，生成标签. shape: (10,)
    X, y = x.reshape(-1, 1), y.reshape(-1, 1)  # 变换数据的维度. shape: (10,1)

    return X, y
