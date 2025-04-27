import torch
import numpy as np
from typing import Dict, Any


class Optimizer:
    """优化器基类"""

    def __init__(self, params, lr=0.001, tol=1e-6, with_sgd=False):
        """

        :param params: 待优化参数.
        """
        self.updaters = None
        self.direction = None
        self.params = params
        self.tol = tol
        self.lr = lr
        self.with_sgd = with_sgd

        self.sk = 's'
        self.s = torch.zeros_like(params)
        self.mk = 'm'
        self.m = torch.zeros_like(params)
        self.states = {self.mk: self.m, self.sk: self.s}

    def grad_model(self, model):
        X = model.X[model.batch_indices] if self.with_sgd else model.X
        y = model.y[model.batch_indices] if self.with_sgd else model.y
        loss = model.criterion.forward(y_predict=model.forward(X), y_true=y)
        self.grad_loss(loss)

    def grad(self, loss):
        self.params.grad = torch.autograd.grad(loss, self.params, create_graph=True)[0]

    def update(self):
        for updater in self.updaters:
            updater.update(self.states)
        self.m = self.states.get(self.mk)
        self.s = self.states.get(self.sk)

    def grad_loss(self, loss):
        """
        计算梯度.
        :param loss: 损失值.
        :return:
        """
        self.grad(loss)
        self.direction = self.params.grad

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

    def optimize(self, loss):
        self.zero_grad()  # 清空梯度.
        self.grad_loss(loss)
        self.step()  # 更新参数.

    def optimize_model(self, model):
        self.zero_grad()  # 清空梯度.
        self.grad_model(model)
        self.step()  # 更新参数.


class GDOptimizer(Optimizer):
    """梯度下降法"""

    def __init__(self, params, lr, with_sgd=False):
        super().__init__(params=params, lr=lr, with_sgd=with_sgd)


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


class MomentumOptimizer(Optimizer):
    """
    动量法优化器.
    """

    def __init__(self, params, lr=0.01, momentum=0.9, with_sgd=False, version=0):
        super().__init__(params=params, lr=lr, with_sgd=with_sgd)
        self.momentum = momentum
        self.m = torch.zeros_like(params)  # 动量.
        if version in [0, 1, 2]:
            self.alpha = [-self.lr, 1, -1][version]  # 根据实现版本，确定各系数的值.
        else:
            self.alpha = -self.lr  # 根据实现版本，确定各系数的值.
        self.gama = -self.lr / self.alpha

    def grad_loss(self, loss):
        """
        计算梯度.
        :param loss: 损失值.
        :return:
        """
        self.grad(loss)
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


class OneMomentumUpdater:
    def __init__(self, params, mto=1.0, mtt=1.0, correction=False):
        self.params = params
        self.momentum_one = mto
        self.momentum_two = mtt
        self.mk = 'm'
        self.correction = correction
        self.t = 1

    def update(self, states: Dict[str, Any]):
        m = self.momentum_one * states.get(self.mk) + self.momentum_two * self.params.grad
        if self.correction:
            m = m / (1 - self.momentum_one ** self.t)
            self.t += 1
        states.update({self.mk: m})


class TwoMomentumUpdater:
    def __init__(self, params, dfo=1.0, dft=1.0, correction=False):
        self.params = params
        self.delay_factor_one = dfo
        self.delay_factor_two = dft
        self.sk = 's'
        self.correction = correction
        self.t = 1

    def update(self, states: Dict[str, Any]):
        s = self.delay_factor_one * states.get(self.sk) + self.delay_factor_two * self.params.grad ** 2
        if self.correction:
            s = s / (1 - self.delay_factor_one ** self.t)
            self.t += 1
        states.update({self.sk: s})


class AdaGrad(Optimizer):
    """
    梯度下降法.
    """

    def __init__(self, params, lr=0.01):
        super().__init__(params, lr=lr)
        self.updaters = [TwoMomentumUpdater(params=params, dfo=1.0, dft=1.0)]
        self.eps = 1e-10

    def grad_loss(self, loss):
        self.grad(loss)
        self.update()
        self.direction = self.params.grad / torch.sqrt(self.s + self.eps)  # 参数更新.


class RMSProp(AdaGrad):
    """
    梯度下降法.
    """

    def __init__(self, params, lr=0.01, delay_factor=0.9):
        super().__init__(params, lr)
        self.delay_factor = delay_factor
        self.updaters = [TwoMomentumUpdater(params=params, dfo=delay_factor, dft=1 - delay_factor)]


class AdaDelta(RMSProp):
    def __init__(self, params, lr=1, delay_factor=0.9):
        super().__init__(params, lr=lr)
        self.updaters = [TwoMomentumUpdater(params=params, dfo=delay_factor, dft=1 - delay_factor)]
        self.delta_square = torch.zeros_like(params)

    def grad_loss(self, loss):
        self.grad(loss)
        self.update()
        delta = torch.sqrt(self.delta_square + self.eps) / torch.sqrt(self.s + self.eps) * self.params.grad
        self.delta_square = self.delay_factor * self.delta_square + (1 - self.delay_factor) * delta * delta
        self.direction = delta


class Adam(AdaGrad):
    """
    梯度下降法.
    """

    def __init__(self, params, lr=0.01, momentum=0.9, delay_factor=0.9):
        super().__init__(params, lr=lr)
        self.updaters = [OneMomentumUpdater(params=params, mto=momentum, mtt=1 - momentum, correction=True),
                         TwoMomentumUpdater(params=params, dfo=delay_factor, dft=1 - delay_factor, correction=True)]

    def grad_loss(self, loss):
        self.grad(loss)
        self.update()
        self.direction = self.m / torch.sqrt(self.s + self.eps)


class NAG(MomentumOptimizer):
    """
    梯度下降法.
    """

    def __init__(self, params, lr=0.01, momentum=0.9, version=0):
        super().__init__(params=params, lr=lr, momentum=momentum)
        if version in [0, 1, 2]:
            self.alpha = [-self.lr, -1, 1][version]
            self.beta = -self.lr / self.alpha
            self.gama = self.beta
        elif version == 3:
            self.alpha = 1
            self.beta = 1
            self.gama = -self.lr
        else:
            self.alpha = -self.lr
            self.beta = -self.lr / self.alpha
            self.gama = self.beta

    def grad_loss(self, loss):
        old_params_data = self.params.data  # 临时存储data的值.
        self.params.data = self.params.data + self.beta * self.momentum * self.m  # 下一步参数的值.
        self.grad(loss)
        self.m = self.momentum * self.m + self.alpha * self.params.grad
        self.params.data = old_params_data
        self.direction = self.m

    def step(self):
        with torch.no_grad():
            self.params.data += self.gama * self.direction
