import math

import torch
import numpy as np


class BaseOptimizer:
    """优化器基类"""

    def __init__(self, params, lr=0.001, tol=1e-8, with_sgd=False):
        """

        :param params: 待优化参数.
        """
        self.updaters = None
        self.direction = None
        self.params = params
        self.tol = tol
        self.lr = lr
        self.adaptive_lr = lr
        self.with_sgd = with_sgd
        self.t = 0
        self.name = self.__class__.__name__

    def grad_model(self, model):
        X = model.X[model.batch_indices] if self.with_sgd else model.X
        y = model.y[model.batch_indices] if self.with_sgd else model.y
        loss = model.criterion.forward(y_predict=model.forward(X), y_true=y)
        self.grad_loss(loss)

    def grad(self, loss):
        loss.backward()
        # self.params.grad = torch.autograd.grad(loss, self.params, create_graph=True)[0]
        self.t += 1

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
            self.params.data -= self.adaptive_lr * self.direction

    def optimize(self, loss):
        self.zero_grad()  # 清空梯度.
        self.grad_loss(loss)
        self.step()  # 更新参数.

    def optimize_model(self, model):
        self.zero_grad()  # 清空梯度.
        self.grad_model(model)
        self.step()  # 更新参数.


class SystemOptimizer(BaseOptimizer):
    def __init__(self, optimizer):
        params = optimizer.param_groups[0].get('params')[0]
        super().__init__(params)
        self.optimizer = optimizer
        self.lr = optimizer.defaults.get('lr')
        self.name = optimizer.__class__.__name__

    def grad_loss(self, loss):
        """
        计算梯度.
        :param loss: 损失值.
        :return:
        """
        self.grad(loss)

    def step(self):
        """
        顺序执行每个更新器，并应用更新策略.
        :return:
        """
        self.optimizer.step()


class BaseMomentumOptimizer(BaseOptimizer):
    """
    动量优化基类.
    """

    def __init__(self, params, lr=0.01, eps=1e-8, momentum_one=1.0, momentum_two=1.0,
                 decay_factor_one=1.0, decay_factor_two=1.0, with_sgd=False):
        super().__init__(params, lr=lr, with_sgd=with_sgd)
        self.direction = None
        self.momentum_one = momentum_one
        self.momentum_two = momentum_two
        self.decay_factor_one = decay_factor_one
        self.decay_factor_two = decay_factor_two
        self.eps = eps
        self.m = torch.zeros_like(params)
        self.m_hat = torch.zeros_like(params)
        self.s = torch.zeros_like(params)
        self.s_hat = torch.zeros_like(params)

    def update_m(self):
        self.m = self.momentum_one * self.m + self.momentum_two * self.params.grad
        return self.m

    def update_s(self):
        self.s = self.decay_factor_one * self.s + self.decay_factor_two * self.params.grad ** 2
        return self.s

    def correct_m(self):
        self.m_hat = self.m / (1 - self.momentum_one ** self.t)
        return self.m_hat

    def correct_s(self):
        self.s_hat = self.s / (1 - self.decay_factor_one ** self.t)
        return self.s_hat

    def get_m_direction(self):
        return self.m_hat / (torch.sqrt(self.s_hat) + self.eps)


class GDOptimizer(BaseOptimizer):
    """梯度下降法"""

    def __init__(self, params, lr, with_sgd=False):
        super().__init__(params=params, lr=lr, with_sgd=with_sgd)


class NewtonOptimizer(BaseOptimizer):
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


class SGDOptimizer(BaseMomentumOptimizer):
    """
    随机梯度下降法：实现与梯度下降相同，不同在于训练时，样本损失不同.
    """

    def __init__(self, params, lr=0.01, gama=0.0):
        """

        :param params:
        :param lr:
        :param gama: 正则化项系数.
        """
        super().__init__(params=params, lr=lr, with_sgd=True)
        self.gama = gama

    def grad_loss(self, loss):
        """
        添加正则化项后的梯度计算.
        :param loss:
        :return:
        """
        self.grad(loss)
        self.params.grad += self.gama * self.params.data
        self.direction = self.params.grad


class SAGOptimizer(BaseOptimizer):

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


class MomentumOptimizer(BaseMomentumOptimizer):
    """
    动量法优化器.
    """

    def __init__(self, params, lr=0.01, momentum=0.9, with_sgd=False, version=0):
        if version in [0, 1, 2]:
            self.alpha = [-lr, 1, -1][version]  # 根据实现版本，确定各系数的值.
        else:
            self.alpha = -lr  # 根据实现版本，确定各系数的值.
        self.gama = -lr / self.alpha
        super().__init__(params=params, lr=lr, momentum_one=momentum, momentum_two=self.alpha, with_sgd=with_sgd)

    def grad_loss(self, loss):
        """
        计算梯度.
        :param loss: 损失值.
        :return:
        """
        self.grad(loss)
        self.update_m()
        self.direction = self.m

    def step(self):
        """
        顺序执行每个更新器，并应用更新策略.
        :return:
        """
        with torch.no_grad():
            if self.params.grad is not None:
                self.params.data += self.gama * self.direction


class AdaGradOptimizer(BaseMomentumOptimizer):
    """
    AdaGrad.
    """

    def __init__(self, params, lr=0.01, eps: float = 1e-10):
        super().__init__(params, lr=lr, eps=eps, decay_factor_one=1.0, decay_factor_two=1.0)

    def grad_loss(self, loss):
        self.grad(loss)
        self.update_s()
        self.direction = self.params.grad / torch.sqrt(self.s + self.eps)


class RMSPropOptimizer(BaseMomentumOptimizer):
    """
    RMSProp.
    """

    def __init__(self, params, lr=1e-2, decay_factor=0.99, eps=1e-8):
        super().__init__(params, lr, eps=eps, decay_factor_one=decay_factor, decay_factor_two=1 - decay_factor)

    def grad_loss(self, loss):
        self.grad(loss)
        self.update_s()
        self.direction = self.params.grad / torch.sqrt(self.s + self.eps)


class AdaDeltaOptimizer(BaseMomentumOptimizer):
    def __init__(self, params, lr=1.0, eps=1e-6, decay_factor=0.9):
        super().__init__(params, lr=lr, eps=eps, decay_factor_one=decay_factor, decay_factor_two=1 - decay_factor)
        self.delta_square = torch.zeros_like(params)

    def grad_loss(self, loss):
        self.grad(loss)
        self.update_s()
        delta = torch.sqrt(self.delta_square + self.eps) / torch.sqrt(self.s + self.eps) * self.params.grad
        self.delta_square = self.decay_factor_one * self.delta_square + self.decay_factor_two * delta * delta
        self.direction = delta


class AdamOptimizer(BaseMomentumOptimizer):
    """
    Adam.
    """

    def __init__(self, params, lr=0.01, eps=1e-8, momentum=0.9, decay_factor=0.999):
        super().__init__(params, lr=lr, eps=eps,
                         momentum_one=momentum, momentum_two=1 - momentum,
                         decay_factor_one=decay_factor, decay_factor_two=1 - decay_factor)

    def grad_loss(self, loss):
        self.grad(loss)
        self.update_m()
        self.update_s()
        self.correct_m()
        self.correct_s()
        self.direction = self.get_m_direction()


class AdamMaxOptimizer(BaseMomentumOptimizer):
    def __init__(self, params, lr=2e-3, eps: float = 1e-8, momentum=0.9, decay_factor=0.999):
        super().__init__(params=params, lr=lr, eps=eps,
                         momentum_one=momentum, momentum_two=1 - momentum,
                         decay_factor_one=decay_factor, decay_factor_two=1 - decay_factor)

    def grad_loss(self, loss):
        self.grad(loss)
        self.update_m()
        self.correct_m()
        norm = torch.abs(self.params.grad) + self.eps
        self.s = torch.max(self.decay_factor_one * self.s, norm)
        self.direction = self.m_hat / self.s


class NAGOptimizer(BaseMomentumOptimizer):
    """
    梯度加速算法.
    """

    def __init__(self, params, lr=1e-3, momentum=0.9, version=0):

        if version in [0, 1, 2, 3]:
            self.alpha = [1, -1, -lr, lr][version]
        else:
            self.alpha = 1
        self.beta = -lr / self.alpha
        self.gama = self.beta
        super().__init__(params=params, lr=lr, momentum_one=momentum, momentum_two=self.alpha)

    def grad_loss(self, loss):
        old_params_data = self.params.data  # 临时存储data的值.
        self.params.data = self.params.data + self.beta * self.momentum_one * self.m  # 下一步参数的值.
        self.grad(loss)
        self.update_m()
        self.params.data = old_params_data

    def step(self):
        with torch.no_grad():
            self.params.data += self.gama * self.m


class NesterovOptimizer(BaseMomentumOptimizer):
    """
    动量加速算法.
    """

    def __init__(self, params, lr=1e-3, momentum=0.9):
        super().__init__(params=params, lr=lr, momentum_one=momentum, momentum_two=1)

    def grad_loss(self, loss):
        self.grad(loss)
        self.update_m()
        self.direction = self.momentum_one * self.m + self.params.grad


class AggMoOptimizer(BaseOptimizer):
    """
    AggMo算法.
    """

    def __init__(self, params, lr=1e-3, momentum_tuple=(0.1, 0.5, 0.9)):
        super().__init__(params, lr)
        self.k = len(momentum_tuple)  # 动量数量.
        self.momentum_s = torch.tensor(momentum_tuple).view(-1, 1)  # 动量系数张量, shape(k,1).
        # +表示形状拼接.
        self.ms = torch.zeros((self.k,) + self.params.shape)  # 动量张量, shape(k,params.shape).

    def grad_loss(self, loss):
        self.grad(loss)
        self.ms = self.momentum_s * self.ms + self.params.grad  # 更新动量张量.
        agg_m = torch.mean(self.ms, dim=0)  # 按照动量系数维度聚合动量.
        self.direction = agg_m


class AdamWOptimizer(BaseMomentumOptimizer):
    def __init__(self, params, lr=2e-3, eps: float = 1e-8, momentum=0.9, decay_factor=0.999,
                 weight_decay=1e-2, alpha=1.0):
        super().__init__(params=params, lr=lr, eps=eps,
                         momentum_one=momentum, momentum_two=1 - momentum,
                         decay_factor_one=decay_factor, decay_factor_two=1 - decay_factor)
        self.weight_decay = weight_decay
        self.alpha = alpha

    def grad_loss(self, loss):
        self.grad(loss)
        self.update_m()
        self.update_s()
        self.correct_m()
        self.correct_s()
        self.direction = self.alpha * self.get_m_direction() + self.weight_decay * self.params.data


class SGDWOptimizer(SGDOptimizer):
    """
    带动量的SGDW.
    momentum=0.0, alpha=1.0.
    """

    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=1e-2, gama=0.01, alpha=1.0):
        """

        :param params:
        :param lr:
        :param momentum:
        :param weight_decay:
        :param gama:
        :param alpha:
        """
        super().__init__(params=params, lr=lr, gama=gama)
        self.weight_decay = weight_decay
        self.momentum_one = momentum
        self.momentum_two = alpha * lr

    def grad_loss(self, loss):
        self.grad(loss)
        self.update_m()

    def step(self):
        self.params.data -= self.m + self.lr * self.gama * self.params.data


class AMSGradOptimizer(AdamOptimizer):

    def grad_loss(self, loss):
        self.grad(loss)
        self.update_m()
        self.correct_m()
        self.update_s()
        self.s_hat = torch.max(self.s_hat, self.s)
        correct_s_hat = self.s_hat / (1 - self.decay_factor_one ** self.t)
        self.direction = self.m_hat / (torch.sqrt(correct_s_hat) + self.eps)


class AdafactorOptimizer(BaseMomentumOptimizer):

    def __init__(self, params, lr=0.01, eps=1e-30, eps2=1e-3, momentum=0.0, decay_factor_exp=-0.8, d=1.0):
        super().__init__(params, lr=lr, eps=eps,
                         momentum_one=momentum, momentum_two=1 - momentum,
                         decay_factor_one=decay_factor_exp, decay_factor_two=1 - decay_factor_exp)
        self.eps2 = eps2
        self.d = d
        self.decay_factor_exp = decay_factor_exp
        if self.params.dim() > 1:
            self.row = torch.zeros(size=(self.params.size(0), 1))
            self.col = torch.zeros(size=(1, self.params.size(1)))

    def grad_loss(self, loss):
        self.grad(loss)
        self.update_m()
        self.decay_factor_two = self.t ** self.decay_factor_exp
        self.decay_factor_one = 1 - self.decay_factor_two

        if self.params.grad.dim() > 1:  # 矩阵形式.
            self.grad_loss_matrix()
        else:  # 向量形式.
            self.update_s()
        u = self.m / (torch.sqrt(self.s) + self.eps)
        rms_u = u.norm(2).item() / u.numel() ** 0.5
        self.direction = u / max(1.0, rms_u / self.d)

        rms_params = self.params.norm(2).item() / self.params.numel() ** 0.5
        rho_t = min(self.lr, 1 / self.t ** 0.5)
        self.adaptive_lr = max(self.eps2, rms_params) * rho_t

    def grad_loss_matrix(self):

        r = self.params.grad.size(0)
        c = self.params.grad.size(-1)
        one_row = torch.ones((1, r))
        one_col = torch.ones((c, 1))

        grad = self.params.grad
        self.row = self.decay_factor_one * self.row + self.decay_factor_two * torch.matmul(grad * grad, one_col)
        self.col = self.decay_factor_one * self.col + self.decay_factor_two * torch.matmul(one_row, grad * grad)
        self.s = torch.matmul(self.row, self.col / torch.matmul(one_row, self.row))


class RAdamOptimizer(AdamOptimizer):
    def __init__(self, params, lr=1e-3, eps=1e-8, momentum=0.9, decay_factor=0.999):
        super().__init__(params, lr=lr, eps=eps, momentum=momentum, decay_factor=decay_factor)
        self.rho_inf = 2 / (1 - decay_factor) - 1

    def grad_loss(self, loss):
        self.grad(loss)
        self.update_m()
        self.update_s()
        self.correct_m()
        a = self.decay_factor_one ** self.t
        rho_t = self.rho_inf - 2 * self.t * a / (1 - a)
        if rho_t > 5:  # 延迟修正.
            self.correct_s()
            b = (rho_t - 4) * (rho_t - 2) * self.rho_inf
            c = (self.rho_inf - 4) * (self.rho_inf - 2) * rho_t
            r_t = math.sqrt(b / c)
            self.direction = r_t * self.get_m_direction()
        else:
            self.direction = self.m_hat


class NadamOptimizer(AdamOptimizer):
    def __init__(self, params, lr=1e-3, eps=1e-8, momentum=0.9, decay_factor=0.999, momentum_decay=4e-3):
        super().__init__(params, lr=lr, eps=eps, momentum=momentum, decay_factor=decay_factor)
        self.momentum_decay = momentum_decay
        self.mu_product = 1.0

    def grad_loss(self, loss):
        self.grad(loss)
        self.update_m()
        self.update_s()

        mu = self.momentum_one * (1.0 - 0.5 * (0.96 ** (self.t * self.momentum_decay)))
        mu_next = self.momentum_one * (1.0 - 0.5 * (0.96 ** ((self.t + 1) * self.momentum_decay)))
        self.mu_product *= mu
        self.m_hat = self.m / (1 - self.mu_product * mu_next)

        self.correct_s()
        grad_hat = self.params.grad / (1 - self.mu_product)

        m_bar = mu_next * self.m_hat + (1 - mu) * grad_hat
        self.direction = m_bar / (torch.sqrt(self.s_hat) + self.eps)


class QHAdamOptimizer(AdamOptimizer):
    def __init__(self, params, lr=1e-3, eps=1e-8, momentum=0.9, decay_factor=0.999, v1=0.7, v2=0.7):
        super().__init__(params, lr=lr, eps=eps, momentum=momentum, decay_factor=decay_factor)
        self.v1 = v1
        self.v2 = v2

    def grad_loss(self, loss):
        self.grad(loss)
        self.update_m()
        self.update_s()
        self.correct_m()
        self.correct_s()
        grad = self.params.grad
        numerator = self.v1 * self.m_hat + (1 - self.v1) * self.params.grad
        denominator = self.v2 * self.s_hat + (1 - self.v2) * grad * grad
        self.direction = numerator / (torch.sqrt(denominator) + self.eps)


class SWATSOptimizer(AdamOptimizer):
    def __init__(self, params, lr=1e-3, eps=1e-8, momentum=0.9, decay_factor=0.999, e=1e-1, sim_e=0):
        super().__init__(params, lr=lr, eps=eps, momentum=momentum, decay_factor=decay_factor)
        self.sgd = 'SGD'
        self.adam = 'Adam'
        self.phase = self.adam
        self.v = torch.zeros_like(self.params)
        self.delta = 0
        self.e = e
        self.sim_e = sim_e

    def grad_loss(self, loss):
        self.grad(loss)  # 梯度计算.
        if self.phase == self.adam:  # Adam阶段.
            self.update_m()  # 更新梯度一阶矩.
            self.update_s()  # 更新梯度二阶矩.
            self.correct_m()  # 修正梯度一阶矩.
            self.correct_s()  # 修正梯度二阶矩.
            p = -self.lr * self.get_m_direction()
            self.params.data += p  # 参数更新.
            sim = torch.sum(p * self.params.grad).item()
            if abs(sim) > self.sim_e:
                gama = - torch.sum(p * p) / sim
                self.delta = self.decay_factor_one * self.delta + self.decay_factor_two * gama
                yita = self.delta / (1 - self.decay_factor_one ** self.t)  # SGD学习率.
                if self.t > 1 and abs(yita - gama) < self.e:  # 阶段切换.
                    self.phase = self.sgd
                    print(f'{self.adam}切换到{self.sgd}: t={self.t}')
                    self.adaptive_lr = (1 - self.momentum_one) * yita  # 设置SGD的自适应学习率.
        else:  # SGD阶段.
            self.v = self.momentum_one * self.v + self.params.grad  # 更新SGD的梯度一阶矩.
            self.params.data -= self.adaptive_lr * self.v  # 参数更新.

    def step(self):
        """
        这里需要覆盖父类step方法实现，在grad_loss方法中进行参数更新.
        :return:
        """
        pass
