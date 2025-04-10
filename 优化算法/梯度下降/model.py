import numpy as np
import torch
from torch import Tensor


class VirtualModel:
    """
    虚拟模型.
    """

    def __init__(self, loss, optimizer) -> None:
        self.loss = loss
        self.optimizer = optimizer
        self.iteration_n = 0
        self.params = self.optimizer.params  # 取参数列表中的第一个张量
        self.points = []

    def fit(self, max_iters=10000):
        loss_history = []
        for epoch in range(max_iters):
            self.points.append(self.params.clone().detach())  # 记录参数每次迭代的值.
            loss = self.loss.forward(self.params)  # 计算损失.
            loss.backward()  # 自动对self.params求梯度
            self.optimizer.step()  # 参数更新.
            loss_history.append(loss.item())
            if self.iteration_n > 0 and abs(loss_history[-1] - loss_history[-2]) < self.optimizer.tol:
                break
            self.iteration_n += 1
        print(f"learning rate {self.optimizer.lr}, Converged at iteration {self.iteration_n}, "
              f"Optimized parameters: {np.round(self.params.detach().numpy(), 3)}, "
              f"Final loss: {loss_history[-1]}")


class LinearModel:

    def __init__(self, in_features: int, out_features: int, loss, optimizer) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.loss = loss
        self.optimizer = optimizer
        self.iteration_n = 0
        self.weights = self.optimizer.params
        self.points = []
        self.loss_history = []

    def _add_bias_term(self, X: Tensor) -> Tensor:
        """为输入数据添加全 1 的偏置列"""
        batch_size = X.shape[0]
        ones = torch.ones((batch_size, 1))  # 添加的常数列
        return torch.cat([X, ones], dim=1)  # 沿特征维度拼接

    def forward(self, X: Tensor) -> Tensor:
        X_augmented = self._add_bias_term(X)  # 添加偏置列
        return X_augmented @ self.weights.reshape(-1, 1)  # 矩阵乘法

    def fit(self, X, y, epochs=10000):
        for epoch in range(epochs):
            self.points.append(self.weights.clone().detach())
            y_hat = self.forward(X)
            loss = self.loss.forward(y_predict=y_hat, y_true=y)
            self.loss_history.append(loss.item())  # 记录损失.
            loss.backward()  # 计算梯度.
            self.optimizer.step()  # 参数更新.
            # 检查收敛条件
            if self.iteration_n > 0 and abs(self.loss_history[-1] - self.loss_history[-2]) < self.optimizer.tol:
                break
            self.iteration_n += 1
        print(f"learning rate {self.optimizer.lr}, Converged at iteration {self.iteration_n}, "
              f"Optimized parameters: {np.round(self.weights.detach().numpy(), 3)}, "
              f"Final loss: {self.loss_history[-1]}")
