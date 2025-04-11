import torch
import numpy as np


class GradientDescent:
    """
    梯度下降法.
    """

    def __init__(self, loss, params, lr=0.01, tol=1e-6):
        """
        初始化方法.
        :param params: 参数列表（可以多组参数）.
        :param lr: 学习率.
        :param tol: 误差阈值.
        """
        self.lr = lr
        self.tol = tol
        self.params = params
        self.loss = loss

    def step(self, loss):
        """
        参数更新步骤.
        :return:
        """
        grad = torch.autograd.grad(loss, self.params, create_graph=True)[0]  # 手动计算梯度.
        self.params.data -= self.lr * grad.data  # 参数更新.

        return loss


class StochasticGradientDescent:
    """
    随机梯度下降法：实现与梯度下降相同，不同在于训练时，样本损失不同.
    """

    def __init__(self, loss, params, lr=0.01, tol=1e-6):
        self.lr = lr
        self.tol = tol
        self.params = params
        self.loss = loss

    def step(self):
        if self.params.grad is not None:
            self.params.data -= self.lr * self.params.grad.data  # 自定义更新规则
            self.params.grad.zero_()


class StochasticAverageGradient:
    def __init__(self, params, lr=0.01, tol=1e-6):
        self.lr = lr
        self.tol = tol
        self.params = params

    def step(self):
        if self.params.grad is not None:
            self.params.data -= self.lr * self.params.grad.data  # 自定义更新规则
            self.params.grad.zero_()


class NewtonGradient:
    def __init__(self, loss, params, lr=0.01, tol=1e-6):
        self.lr = lr
        self.tol = tol
        self.params = params
        self.loss = loss

    def step(self, loss):
        grad = torch.autograd.grad(loss, self.params, create_graph=True)[0]  # 手动计算梯度.
        hessian_matrix = self.compute_hessian(grad)  # 计算hessian矩阵.
        # from torch.autograd.functional import hessian  # 导入hessian函数.
        # hessian_matrix = hessian(self.loss.forward, self.params)  # 使用内置的hessian.
        hessian_matrix += 1e-3 * np.eye(len(self.params))  # 正则化
        delta = np.linalg.solve(hessian_matrix, grad.data.numpy())  # H逆与梯度向量的乘积.
        self.params.data -= self.lr * torch.from_numpy(delta).float()

    def compute_hessian(self, grads):
        # 计算Hessian矩阵
        hessian_arr = []
        for g in grads:
            g2 = torch.autograd.grad(g, self.params, retain_graph=True)[0]
            hessian_arr.append(g2.data.numpy())

        return np.array(hessian_arr)

# class NewtonGradient:
#     def __init__(self, params, lr=0.01, tol=1e-6):
#         self.lr = lr
#         self.tol = tol
#         self.params = params
#
#     def step(self):
#         if self.params.grad is not None:
#             hessian_matrix = self.compute_hessian()
#             self.params.data -= self.lr * self.params.grad.data  # 自定义更新规则
#             self.params.grad.zero_()
#
#     def compute_hessian(self):
#         # 计算Hessian矩阵
#         hessian_arr = []
#         grad = self.params.grad.detach().clone().requires_grad_(True)  # 重建计算图
#         for g in grad:
#             # g2 = torch.autograd.grad(g, self.params, retain_graph=True,allow_unused=True)[0]
#             g2 = torch.autograd.grad(
#                 outputs=g,
#                 inputs=self.params,
#                 retain_graph=True,
#                 allow_unused=True
#             )[0]
#             hessian_arr.append(g2.data.numpy())
#
#         return np.array(hessian_arr)
