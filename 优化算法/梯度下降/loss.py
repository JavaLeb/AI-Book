import torch
from torch import Tensor
import numpy as np


class Loss:
    """
    损失函数f(x,y) = x1^2 + 2 * x2^2.
    """

    def __init__(self) -> None:
        self.c = torch.tensor([1.0, 2.0])  # 确保类型为float

    def forward(self, params: Tensor) -> Tensor:
        return torch.sum(self.c * params ** 2, dim=-1)  # 移除dim参数

    def mesh(self, x, y):
        """
        绘制损失函数等高线.
        :return:
        """

        X, Y = np.meshgrid(x, y)  # x沿着y的维度复制得到X，y沿着x的维度复制得到Y. 让X和Y重叠，重叠位置就是(x,y)坐标.

        points = np.stack([X.ravel(), Y.ravel()], axis=1)  # 网格点变为坐标点(x,y).
        points = torch.tensor(points, dtype=torch.float32)  # 转化为张量.
        Z = self.forward(points)
        Z = Z.reshape(X.shape).numpy()  # 转换为网格数据.

        return X, Y, Z


class MSELoss:
    def __init__(self) -> None:
        pass

    def forward(self, y_predict: Tensor, y_true: Tensor) -> Tensor:
        return torch.mean((y_predict - y_true) ** 2)

    def mesh(self, w, b, X_samples, y_samples):
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

        return W.numpy(), B.numpy(), loss_grid.numpy()
