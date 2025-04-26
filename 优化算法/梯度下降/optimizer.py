import torch
import numpy as np


class Optimizer:
    """优化器基类"""

    def __init__(self, params, lr=0.001, tol=1e-6, with_sgd=False):
        """

        :param params: 待优化参数.
        :param updaters: 更新器.
        """
        self.direction = None
        self.params = params
        self.tol = tol
        self.lr = lr
        self.with_sgd = with_sgd

    def grad_model(self, model):
        X = model.X[model.batch_indices] if self.with_sgd else model.X
        y = model.y[model.batch_indices] if self.with_sgd else model.y
        loss = model.criterion.forward(y_predict=model.forward(X), y_true=y)
        self.grad_loss(loss)

    def grad_loss(self, loss):
        """
        计算梯度.
        :param loss: 损失值.
        :return:
        """
        self.direction = torch.autograd.grad(loss, self.params, create_graph=True)[0]

    def zero_grad(self):
        """
        清空梯度.
        :return:
        """
        if self.params.grad is not None:
            self.params.grad.zero_()

    def step(self):
        """
        顺序执行每个更新器，并应用更新策略.
        :return:
        """
        with torch.no_grad():
            self.params.data -= self.lr * self.direction


class GDOptimizer(Optimizer):
    """梯度下降法"""

    def __init__(self, params, lr, with_sgd=False):
        super().__init__(params=params, lr=lr, with_sgd=with_sgd)


class MomentumOptimizer(Optimizer):
    """
    动量法优化器.
    """

    def __init__(self, params, lr=0.01, momentum=0.9, with_sgd=False, version=0):
        super().__init__(params=params, lr=lr, with_sgd=with_sgd)
        self.momentum = momentum
        self.m = torch.zeros_like(params)  # 动量.
        self.alpha = [-self.lr, 1, -1][version]  # 根据实现版本，确定各系数的值.
        self.gama = -self.lr / self.alpha

    def grad_loss(self, loss):
        """
        计算梯度.
        :param loss: 损失值.
        :return:
        """
        self.params.grad = torch.autograd.grad(loss, self.params, create_graph=True)[0]
        self.m = self.momentum * self.m + self.alpha * self.params.grad  # 动量更新.
        self.direction = self.m

    def step(self):
        """
        顺序执行每个更新器，并应用更新策略.
        :return:
        """
        with torch.no_grad():
            if self.params.grad is not None:
                self.params.data += self.gama * self.direction


class RMSProp(Optimizer):
    """
    梯度下降法.
    """

    def __init__(self, params, lr=0.01, delay_factor=0.9):
        super().__init__(params, lr)
        self.delay_factor = delay_factor
        self.s = torch.zeros_like(params)  # 动量.
        self.error = 1e-10

    def grad_loss(self, loss):
        self.params.grad = torch.autograd.grad(loss, self.params, create_graph=True)[0]  # 手动计算梯度.
        self.s = self.delay_factor * self.s + (1 - self.delay_factor) * self.params.grad ** 2
        self.direction = self.params.grad / torch.sqrt(self.s + self.error)  # 参数更新.


class AdaGrad(Optimizer):
    """
    梯度下降法.
    """

    def __init__(self, params, lr=0.01):
        super().__init__(params, lr)
        self.s = torch.zeros_like(params)
        self.error = 1e-10

    def grad_loss(self, loss):
        self.params.grad = torch.autograd.grad(loss, self.params, create_graph=True)[0]  # 手动计算梯度.
        self.s = self.s + self.params.grad ** 2
        self.direction = self.params.grad / torch.sqrt(self.s + self.error)  # 参数更新.


class NewtonOptimizer(Optimizer):
    """
    牛顿法.
    """

    def __init__(self, params, lr=0.01, with_sgd=False):
        super().__init__(params=params, lr=lr, with_sgd=with_sgd)

    def grad_loss(self, loss):
        self.params.grad = torch.autograd.grad(loss, self.params, create_graph=True)[0]  # 手动计算梯度.
        hessian_matrix = self.compute_hessian()  # 计算hessian矩阵.
        delta = np.linalg.solve(hessian_matrix, self.params.grad.data.numpy())  # H逆与梯度向量的乘积.
        self.direction = torch.from_numpy(delta).float()

    def compute_hessian(self):
        """
        Hessian矩阵计算.
        :return: Hessian矩阵.
        """
        # 使用内置hessian函数计算.
        # from torch.autograd.functional import hessian  # 导入hessian函数.
        # hessian_matrix = hessian(self.loss.forward, self.params)  # 使用内置的hessian.

        # 自定义计算Hessian矩阵.
        hessian_arr = []
        for g in self.params.grad:
            g2 = torch.autograd.grad(g, self.params, retain_graph=True)[0]
            hessian_arr.append(g2.data.numpy())
        hessian_matrix = np.array(hessian_arr)
        # hessian_matrix += 1e-3 * np.eye(len(self.params))  # 正则化

        return hessian_matrix


class SGDOptimizer(Optimizer):
    """
    随机梯度下降法：实现与梯度下降相同，不同在于训练时，样本损失不同.
    """

    def __init__(self, params, lr=0.01, with_sgd=False):
        self.with_sgd = True
        super().__init__(params=params, lr=lr, with_sgd=with_sgd)


class SAGOptimizer(Optimizer):

    def grad_model(self, model):
        y_hat = model.forward(model.X[model.batch_indices])
        mini_loss = model.criterion.forward(y_predict=y_hat,
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

        self.direction = model.average_grads

# class NestAcceleratedGradient(MomentumOptimizer):
#     """
#     梯度下降法.
#     """
#
#     def __init__(self, loss, params, lr=0.01, tol=1e-6, momentum=0.9):
#         super().__init__(loss, params, lr, tol, momentum)
#
#     def step(self, model=None):
#         """
#         动量加速等价形式.
#         :param model:
#         :return:
#         """
#
#         grad = torch.autograd.grad(model.loss, self.params, create_graph=True)[0]  # 手动计算梯度.
#         self.m = self.beta ** 2 * self.m + (self.beta + 1) * grad.data
#         self.params.data -= self.lr * self.m  # 参数更新.
#     # def step(self, model=None):
#     #     """
#     #     对动量加速.
#     #     :param model:
#     #     :return:
#     #     """
#     #
#     #     grad = torch.autograd.grad(model.loss, self.params, create_graph=True)[0]  # 手动计算梯度.
#     #     self.m = self.beta * self.m + grad.data
#     #     self.m = self.beta * self.m + grad.data
#     #     self.params.data -=  self.lr * self.m  # 参数更新.
#
#     # def step(self, model=None):
#     #     old_params_data = self.params.data  # 临时存储data的值.
#     #     # theta_tmp = theta_t + beta * m
#     #     self.params.data = self.params.data + self.beta * self.m  # 下一步参数的值.
#     #     grad = torch.autograd.grad(model.loss, self.params, create_graph=True)[0]  # 手动计算梯度.
#     #     self.m = self.beta * self.m - self.lr * grad.data
#     #     self.params.data = old_params_data + self.m  # 参数更新.
#
#     # def step(self, model=None):
#     #     old_params_data = self.params.data  # 临时存储data的值.
#     #     # theta_tmp = theta_t + beta * m
#     #     self.params.data = self.params.data + self.beta * self.m  # 下一步参数的值.
#     #     grad = torch.autograd.grad(model.loss, self.params, create_graph=True)[0]  # 手动计算梯度.
#     #     self.m = self.beta * self.m + grad.data
#     #     self.params.data = old_params_data - self.lr * self.m  # 参数更新.
#
#

#
#

#
#
# class Adam(RMSProp):
#     """
#     梯度下降法.
#     """
#
#     def __init__(self, loss, params, lr=0.01, tol=1e-6, delay_factor=0.9, momentum=0.9):
#         super().__init__(loss, params, lr, tol, delay_factor)
#         self.momentum = momentum
#         self.m = torch.zeros_like(params)
#         self.t = 0
#
#     def step(self, model=None):
#         grad = torch.autograd.grad(model.loss, self.params, create_graph=True)[0]  # 手动计算梯度.
#         self.m = self.momentum * self.m + (1 - self.momentum) * grad
#         self.scale = self.delay_factor * self.scale + (1 - self.delay_factor) * grad.data ** 2
#
#         modified_m = self.m / (1 - self.momentum ** (self.t + 1))
#         modified_scale = self.scale / (1 - self.delay_factor ** (self.t + 1))
#
#         self.params.data -= self.lr * modified_m / torch.sqrt(modified_scale + self.error)  # 参数更新.
#
#         self.t += 1
