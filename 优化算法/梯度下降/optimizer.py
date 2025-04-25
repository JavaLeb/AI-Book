import torch
import numpy as np


class GradientDescent:
    """
    梯度下降法.
    """

    def __init__(self, params, lr=0.01, tol=1e-6):
        """
        初始化方法.
        :param params: 参数.
        :param lr: 学习率.
        :param tol: 优化停止误差阈值.
        """
        self.params = params
        self.lr = lr
        self.tol = tol
        self.model_flag = False

    def set_params(self, params):
        self.params = params

    def set_lr(self, lr):
        self.lr = lr

    def grad(self, loss):
        """
        计算损失梯度.
        :param loss: 损失.
        :return:
        """
        loss.backward()  # 自动计算梯度.

    def step(self):
        """
        参数更新步骤.
        """
        with torch.no_grad():
            if self.params.grad is not None:
                self.params.data -= self.lr * self.params.grad

    def zero_grad(self):
        """
        清空梯度.
        :return:
        """
        if self.params.grad is not None:
            self.params.grad.zero_()


class NewtonGradient(GradientDescent):
    """
    牛顿法.
    """

    def grad(self, loss):
        self.grads = torch.autograd.grad(loss, self.params, create_graph=True)[0]

    def step(self):
        """
        优化步骤.
        :return:
        """
        hessian_matrix = self.compute_hessian()  # 计算hessian矩阵.
        delta = np.linalg.solve(hessian_matrix, self.grads.data.numpy())  # H逆与梯度向量的乘积.
        self.params.data -= self.lr * torch.from_numpy(delta).float()

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
        for g in self.grads:
            g2 = torch.autograd.grad(g, self.params, retain_graph=True)[0]
            hessian_arr.append(g2.data.numpy())
        hessian_matrix = np.array(hessian_arr)
        # hessian_matrix += 1e-3 * np.eye(len(self.params))  # 正则化

        return hessian_matrix


class StochasticGradientDescent(GradientDescent):
    """
    随机梯度下降法：实现与梯度下降相同，不同在于训练时，样本损失不同.
    """

    def __init__(self, params, lr=0.01, tol=1e-6):
        super().__init__(params, lr, tol)
        self.model_flag = True

    def grad(self, model):
        loss = model.criterion.forward(y_predict=model.forward(model.X[model.batch_indices]),
                                       y_true=model.y[model.batch_indices])
        super().grad(loss)


class StochasticAverageGradient(StochasticGradientDescent):
    def __init__(self, params, lr=0.01, tol=1e-6):
        super().__init__(params, lr, tol)
        self.average_grads = None

    def grad(self, model):
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
        self.average_grads = model.average_grads
        # 更新每个样本梯度.
        model.samples_grads[model.batch_indices] = mini_grad_matrix

    def step(self):
        # 参数更新.
        self.params.data -= self.lr * self.average_grads


class MomentumOptimizer(GradientDescent):
    """
    动量法.
    """

    def __init__(self, params, lr=0.01, tol=1e-6, momentum=0.9):
        super().__init__(params, lr, tol)
        self.beta = momentum  # 一阶矩系数beta.
        self.m = torch.zeros_like(params)  # 动量.

    def step(self):
        self.m = self.beta * self.m + self.params.grad  # 动量更新.
        self.params.data += - self.lr * self.m  # 参数更新.


class Momentum2Optimizer(MomentumOptimizer):
    def step(self, model=None):
        self.m = self.beta * self.m - self.lr * self.params.grad  # 动量更新.
        self.params.data += self.m  # 参数更新.


class Momentum3Optimizer(MomentumOptimizer):
    def step(self, model=None):
        self.m = self.beta * self.m - self.params.grad  # 动量更新.
        self.params.data += self.lr * self.m  # 参数更新.


class NestAcceleratedGradient(MomentumOptimizer):
    """
    梯度下降法.
    """

    def __init__(self, loss, params, lr=0.01, tol=1e-6, momentum=0.9):
        super().__init__(loss, params, lr, tol, momentum)

    def step(self, model=None):
        """
        动量加速等价形式.
        :param model:
        :return:
        """

        grad = torch.autograd.grad(model.loss, self.params, create_graph=True)[0]  # 手动计算梯度.
        self.m = self.beta ** 2 * self.m + (self.beta + 1) * grad.data
        self.params.data -= self.lr * self.m  # 参数更新.
    # def step(self, model=None):
    #     """
    #     对动量加速.
    #     :param model:
    #     :return:
    #     """
    #
    #     grad = torch.autograd.grad(model.loss, self.params, create_graph=True)[0]  # 手动计算梯度.
    #     self.m = self.beta * self.m + grad.data
    #     self.m = self.beta * self.m + grad.data
    #     self.params.data -=  self.lr * self.m  # 参数更新.

    # def step(self, model=None):
    #     old_params_data = self.params.data  # 临时存储data的值.
    #     # theta_tmp = theta_t + beta * m
    #     self.params.data = self.params.data + self.beta * self.m  # 下一步参数的值.
    #     grad = torch.autograd.grad(model.loss, self.params, create_graph=True)[0]  # 手动计算梯度.
    #     self.m = self.beta * self.m - self.lr * grad.data
    #     self.params.data = old_params_data + self.m  # 参数更新.

    # def step(self, model=None):
    #     old_params_data = self.params.data  # 临时存储data的值.
    #     # theta_tmp = theta_t + beta * m
    #     self.params.data = self.params.data + self.beta * self.m  # 下一步参数的值.
    #     grad = torch.autograd.grad(model.loss, self.params, create_graph=True)[0]  # 手动计算梯度.
    #     self.m = self.beta * self.m + grad.data
    #     self.params.data = old_params_data - self.lr * self.m  # 参数更新.


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
