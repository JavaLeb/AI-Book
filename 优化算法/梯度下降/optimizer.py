import torch
import numpy as np


class GradientDescent:
    """
    梯度下降法.
    """

    def __init__(self, loss, params, lr=0.01, tol=1e-6):
        """
        初始化方法.
        :param loss: 损失，例如MSELoss().
        :param params: 参数列表（可以多组参数）.
        :param lr: 学习率.
        :param tol: 误差阈值.
        """

        self.loss = loss
        self.lr = lr
        self.tol = tol
        self.params = params

    def step(self, model=None):
        """
        参数更新步骤.
        :param loss: 损失值.
        """
        grad = torch.autograd.grad(model.loss, self.params, create_graph=True)[0]  # 手动计算梯度.
        self.params.data -= self.lr * grad.data  # 参数更新.


class NewtonGradient(GradientDescent):
    def __init__(self, loss, params, lr=0.01, tol=1e-6):
        super().__init__(loss, params, lr, tol)

    def step(self, model=None):
        """
        优化步骤.
        :return:
        """
        grad = torch.autograd.grad(model.loss, self.params, create_graph=True)[0]  # 手动计算梯度.
        hessian_matrix = self.compute_hessian(grad)  # 计算hessian矩阵.
        # from torch.autograd.functional import hessian  # 导入hessian函数.
        # hessian_matrix = hessian(self.loss.forward, self.params)  # 使用内置的hessian.
        delta = np.linalg.solve(hessian_matrix, grad.data.numpy())  # H逆与梯度向量的乘积.
        self.params.data -= self.lr * torch.from_numpy(delta).float()

    def compute_hessian(self, grads):
        """
        Hessian矩阵计算.
        :param grads: 梯度向量
        :return: Hessian矩阵.
        """
        # 计算Hessian矩阵
        hessian_arr = []
        for g in grads:
            g2 = torch.autograd.grad(g, self.params, retain_graph=True)[0]
            hessian_arr.append(g2.data.numpy())
        hessian_matrix = np.array(hessian_arr)
        # hessian_matrix += 1e-3 * np.eye(len(self.params))  # 正则化

        return hessian_matrix


class StochasticGradientDescent(GradientDescent):
    """
    随机梯度下降法：实现与梯度下降相同，不同在于训练时，样本损失不同.
    """

    def __init__(self, loss, params, lr=0.01, tol=1e-6):
        super().__init__(loss, params, lr, tol)

    def step(self, model=None):
        mini_loss = self.loss.forward(y_predict=model.forward(model.X[model.batch_indices]),
                                      y_true=model.y[model.batch_indices])
        mini_loss.backward()  # 计算梯度.
        if self.params.grad is not None:
            self.params.data -= self.lr * self.params.grad.data  # 自定义更新规则
            self.params.grad.zero_()


class StochasticAverageGradient(GradientDescent):
    def __init__(self, loss, params, lr=0.01, tol=1e-6):
        super().__init__(loss, params, lr, tol)

    def step(self, model=None):
        y_hat = model.forward(model.X[model.batch_indices])
        mini_loss = self.loss.forward(y_predict=y_hat,
                                      y_true=model.y[model.batch_indices], reduction=None)
        # 计算小批量样本梯度矩阵.
        grads_per_sample = []
        for i in range(len(mini_loss)):
            grad = torch.autograd.grad(mini_loss[i], self.params, retain_graph=True)[0]
            grads_per_sample.append(grad)
        mini_grad_matrix = torch.stack(grads_per_sample, dim=0)
        # 更新平均梯度.
        mini_grad_new_sum = torch.sum(mini_grad_matrix, dim=0)
        mini_grad_old_sum = torch.sum(model.samples_grads[model.batch_indices], dim=0)
        model.average_grads += (mini_grad_new_sum - mini_grad_old_sum) / model.n_samples
        # 更新每个样本梯度.
        model.samples_grads[model.batch_indices] = mini_grad_matrix
        # 参数更新.
        self.params.data -= self.lr * model.average_grads


class MomentumOptimizer(GradientDescent):
    """
    梯度下降法.
    """

    def __init__(self, loss, params, lr=0.01, tol=1e-6, momentum=0.9):
        super().__init__(loss, params, lr, tol)
        self.momentum = momentum
        self.m = torch.zeros_like(params)

    def step(self, model=None):
        grad = torch.autograd.grad(model.loss, self.params, create_graph=True)[0]  # 手动计算梯度.
        self.m = self.momentum * self.m + grad.data  # 动量更新.
        self.params.data += - self.lr * self.m  # 参数更新.


class NestAcceleratedGradient(MomentumOptimizer):
    """
    梯度下降法.
    """

    def __init__(self, loss, params, lr=0.01, tol=1e-6, momentum=0.9):
        super().__init__(loss, params, lr, tol, momentum)

    def step(self, model=None):
        old_params_data = self.params.data  # 临时存储data的值.
        self.params.data = self.params.data - self.lr * self.momentum * self.m  # 下一步参数的值.
        grad = torch.autograd.grad(model.loss, self.params, create_graph=True)[0]  # 手动计算梯度.
        self.m = self.momentum * self.m + grad.data
        self.params.data = old_params_data - self.lr * self.m  # 参数更新.


class AdaGrad(GradientDescent):
    """
    梯度下降法.
    """

    def __init__(self, loss, params, lr=0.01, tol=1e-6):
        super().__init__(loss, params, lr, tol)
        self.scale = torch.zeros_like(params)
        self.error = 1e-10

    def step(self, model=None):
        grad = torch.autograd.grad(model.loss, self.params, create_graph=True)[0]  # 手动计算梯度.
        self.scale += grad.data ** 2
        self.params.data -= self.lr * grad.data / torch.sqrt(self.scale + self.error)  # 参数更新.


class RMSProp(AdaGrad):
    """
    梯度下降法.
    """

    def __init__(self, loss, params, lr=0.01, tol=1e-6, delay_factor=0.9):
        super().__init__(loss, params, lr, tol)
        self.delay_factor = delay_factor

    def step(self, model=None):
        grad = torch.autograd.grad(model.loss, self.params, create_graph=True)[0]  # 手动计算梯度.
        self.scale = self.delay_factor * self.scale + (1 - self.delay_factor) * grad.data ** 2
        self.params.data -= self.lr * grad.data / torch.sqrt(self.scale + self.error)  # 参数更新.


class Adam(RMSProp):
    """
    梯度下降法.
    """

    def __init__(self, loss, params, lr=0.01, tol=1e-6, delay_factor=0.9, momentum=0.9):
        super().__init__(loss, params, lr, tol, delay_factor)
        self.momentum = momentum
        self.m = torch.zeros_like(params)
        self.t = 0

    def step(self, model=None):
        grad = torch.autograd.grad(model.loss, self.params, create_graph=True)[0]  # 手动计算梯度.
        self.m = self.momentum * self.m + (1 - self.momentum) * grad
        self.scale = self.delay_factor * self.scale + (1 - self.delay_factor) * grad.data ** 2

        modified_m = self.m / (1 - self.momentum ** (self.t + 1))
        modified_scale = self.scale / (1 - self.delay_factor ** (self.t + 1))

        self.params.data -= self.lr * modified_m / torch.sqrt(modified_scale + self.error)  # 参数更新.

        self.t += 1
