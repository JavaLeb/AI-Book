import torch
from torch import Tensor
import numpy as np
from data import generate_data
from stochastic_gradient_descent import StochasticGradientDescent
from loss import MSELoss


class LinearModel:
    def __init__(self, in_features: int, out_features: int, loss, optimizer) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.loss = loss
        self.optimizer = optimizer
        self.weight = self.optimizer.params[0]
        self.bias = self.optimizer.params[1]

    def forward(self, X: Tensor) -> Tensor:
        return self.weight * X + self.bias

    def fit(self, X, y, epochs=10000, batch_size=10, shuffled=False):

        n_samples, n_features = X.shape  # 获取样本数和特征数.

        # 训练循环
        iteration_n = 0
        loss_history = []
        epoch_n = 0
        for epoch in range(epochs):
            if shuffled:
                shuffled_indices = np.random.permutation(n_samples)  # 样本序号打乱.
            else:
                shuffled_indices = np.arange(n_samples)  # 固定顺序
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = shuffled_indices[start_idx:end_idx]
                batch_X = X[batch_indices]
                batch_y = y[batch_indices]
                y_hat = self.forward(batch_X)
                loss = self.loss.forward(y_predict=y_hat, y_true=batch_y)
                loss_history.append(loss.item())  # 记录损失.
                loss.backward()  # 计算梯度.
                self.optimizer.step()  # 参数更新.
                # 检查收敛条件
                if iteration_n > 0 and abs(loss_history[-1] - loss_history[-2]) < self.optimizer.tol:
                    break
                iteration_n += 1
            epoch_n = epoch
        print(f"learning rate {self.optimizer.lr}, Converged at epoch {epoch_n}, iteration {iteration_n}, "
              f"Optimized parameters: {np.round(self.weight.detach().numpy(), 3)}, {np.round(self.bias.detach().numpy(), 3)}, "
              f"Final loss: {loss_history[-1]}")


if __name__ == '__main__':
    # 独立初始化参数（确保形状正确）
    X, y = generate_data()
    weight = torch.tensor([[10.0]], requires_grad=True)  # shape: [out=1, in=1]
    bias = torch.tensor([10.0], requires_grad=True)  # shape: [out=1]

    model = LinearModel(
        in_features=1,
        out_features=1,
        loss=MSELoss(),
        optimizer=StochasticGradientDescent(params=[weight, bias], lr=0.01)  # 直接传递模型参数
    )

    model.fit(X, y)
