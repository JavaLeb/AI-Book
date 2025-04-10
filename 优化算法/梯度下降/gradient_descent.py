import torch
from model import VirtualModel
from loss import SquareFunction


class GradientDescent:
    """
    梯度下降法.
    """

    def __init__(self, params, lr=0.01, tol=1e-6):
        """
        初始化方法.
        :param params: 参数列表（可以多组参数）.
        :param lr: 学习率.
        :param tol: 误差阈值.
        """
        self.lr = lr
        self.tol = tol
        self.params = params

    def step(self):
        """
        参数更新步骤.
        :return:
        """
        if self.params.grad is not None:
            self.params.data -= self.lr * self.params.grad.data  # 参数更新.
            self.params.grad.zero_()  # 清空梯度.


if __name__ == '__main__':
    initial_param = torch.tensor([10.0, 10.0], requires_grad=True)
    model = VirtualModel(
        loss=SquareFunction(torch.tensor([1.0, 2.0])),
        optimizer=GradientDescent(params=initial_param, lr=0.1)
    )
    model.fit()
