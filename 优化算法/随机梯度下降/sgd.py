import torch
from torch import Tensor
import numpy as np


class MSELoss:
    def __init__(self) -> None:
        pass

    def forward(self, y_predict: Tensor, y_true: Tensor) -> Tensor:
        power = (y_predict - y_true) ** 2
        m = torch.mean(power)
        return m


class SGD:
    def __init__(self, params, lr=0.01, tol=1e-6):
        self.lr = lr
        self.tol = tol
        self.params = params

    def step(self):
        with torch.no_grad():
            for param in self.params:
                if param.grad is not None:
                    param.data -= self.lr * param.grad.data  # 自定义更新规则
                    param.grad.zero_()


class LinearModel:
    def __init__(self, in_features: int, out_features: int, loss: MSELoss, optimizer: SGD) -> None:
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


def generate_data():
    torch.manual_seed(42)
    x = torch.linspace(0, 10, 10)  # shape: (10,).
    true_w, true_b = 2.0, 1.0
    y = true_w * x + true_b + torch.randn(x.size()) * 1.5  # shape: (10,)
    X = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    return X, y


if __name__ == '__main__':
    # 独立初始化参数（确保形状正确）
    weight = torch.tensor([[10.0]], requires_grad=True)  # shape: [out=1, in=1]
    bias = torch.tensor([10.0], requires_grad=True)  # shape: [out=1]

    model = LinearModel(
        in_features=1,
        out_features=1,
        loss=MSELoss(),
        optimizer=SGD(params=[weight, bias], lr=0.01)  # 直接传递模型参数
    )

    X, y = generate_data()
    model.fit(X, y)
