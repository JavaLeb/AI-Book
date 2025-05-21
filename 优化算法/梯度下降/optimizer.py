import math
import torch
import numpy as np
import torch_optimizer


class BaseOptimizer:
    """优化器基类"""

    def __init__(self, params, lr=0.001, eps=1e-8, tol=1e-9,
                 tol_grad=1e-7, with_sgd=False, dynamic=False):
        """

        :param params: 待优化参数.
        :param lr: 学习率.
        :param eps: 用于调整分母，使其不为零.
        :param tol: 允许阈值.
        :param tol_grad: 梯度阈值.
        :param with_sgd: 是否支持随机梯度下降.
        :param dynamic: 是否支持动态学习率.
        """
        self.params = params
        self.lr = lr
        self.eps = eps
        self.tol = tol
        self.tol_grad = tol_grad
        self.with_sgd = with_sgd
        self.dynamic = dynamic
        self.step_t = 0  # 迭代时间步数, 它等于当前时间步+1.
        self.adaptive_lr = lr  # 动态学习率.
        self.direction = None  # 迭代方向.
        self.name = self.__class__.__name__  # 类名.
        self.loss = None  # 损失值.
        self.optimized_params = None  # 最优参数.
        self.is_shampoo = False
        self.is_closure = False

    def zero_grad(self):
        """
        清空梯度.
        :return:
        """
        if self.params.grad is not None:
            self.params.grad.zero_()

    def grad(self, loss):
        """
        计算梯度.
        :param loss: 损失值.
        :return:
        """
        # 手工计算梯度.
        self.params.grad = torch.autograd.grad(loss, self.params, create_graph=True)[0]  # newton需要.

    def grad_model(self, model):
        """
        针对模型优化，计算梯度.
        :param model: 模型.
        :return:
        """
        X = model.X[model.batch_indices]
        y = model.y[model.batch_indices]
        loss = model.criterion.forward(y_predict=model.forward(X), y_true=y)  # 计算模型的损失.
        self.grad(loss)  # 计算梯度.

    def step(self):
        """
        顺序执行每个更新器，并应用更新策略.
        :return:
        """
        self.step_t += 1
        if self.dynamic:  # 动态学习率
            self.adaptive_lr = self.dynamic_lr()  # 迭代步长.
        self.direction = self.params.grad  # 迭代方向
        self.update_params()  # 更新参数

    def dynamic_lr(self, alpha=0.5, beta=1.0):
        """
        计算动态学习率.
        :param alpha: 常数.
        :param beta: 常数.
        :return:
        """
        return self.lr / (beta * self.step_t + 1) ** alpha

    def update_params(self):
        """
        参数更新.
        :return:
        """
        with torch.no_grad():
            self.params.data -= self.adaptive_lr * self.direction  # 参数更新.
            self.optimized_params = self.params.data  # 设置最优参数.

    def optimize(self, loss):
        """
        一步完整的优化.
        :param loss: 损失值.
        :return:
        """
        self.zero_grad()  # 清空梯度.
        self.grad(loss)  # 计算梯度.
        self.step()  # 更新参数.

    def optimize_model(self, model):
        """
        针对模型的一步完整的优化.
        :param model: 模型.
        :return:
        """
        self.zero_grad()  # 清空梯度.
        self.grad_model(model)  # 计算梯度.
        self.step()  # 更新参数.


class SystemOptimizer(BaseOptimizer):
    """
    封装了系统内置的优化算法，用于和自定义实现对比优化结果.
    """

    def __init__(self, optimizer):
        """
        初始化方法.
        :param optimizer: 内置优化器.
        """
        params = optimizer.param_groups[0].get('params')[0]  # 获取内置优化器待优化的参数.
        super().__init__(params)
        self.optimizer = optimizer  # 内置优化器.
        self.lr = optimizer.defaults.get('lr')  # 获取内置优化器学习率.
        self.name = optimizer.__class__.__name__  # 内置优化器类名.
        self.is_shampoo = isinstance(optimizer, torch_optimizer.Shampoo)
        self.is_closure = isinstance(self.optimizer, torch.optim.LBFGS)

    def step(self):
        """
        优化步.
        :return:
        """
        self.optimizer.step()  # 一步优化.
        self.optimized_params = self.params.data  # 设置最优参数.


class GDOptimizer(BaseOptimizer):
    """梯度下降法"""


class NewtonOptimizer(BaseOptimizer):
    """
    牛顿法.
    """

    def __init__(self, params, lr=0.01, with_sgd=False):
        super().__init__(params=params, lr=lr, with_sgd=with_sgd)

    def step(self):
        self.step_t += 1
        hessian_matrix = self.compute_hessian()  # 计算hessian矩阵.
        delta = np.linalg.solve(hessian_matrix, self.params.grad.data.numpy())  # H逆与梯度向量的乘积.
        self.direction = torch.from_numpy(delta).float()
        super().update_params()

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


class LBFGSOptimizer(BaseOptimizer):
    def __init__(self, params, lr=0.1, history_size=100):
        super().__init__(params, lr=lr)
        self.history_size = history_size
        self.prev_grad = torch.zeros_like(self.params)
        self.his_vector = []
        self.H_diag = None

    def step(self):
        self.step_t += 1
        grad = self.params.grad  # 梯度.
        q = grad.neg()
        if self.step_t == 1:
            self.H_diag = 1.0
            r = self.H_diag * q
            self.adaptive_lr = min(1.0, 1.0 / grad.abs().sum()) * self.lr
        else:
            g_ch = grad - self.prev_grad  # 梯度变化.
            p_ch = self.adaptive_lr * self.direction  # 参数变化.
            gama = g_ch.dot(p_ch)
            if gama > 0:
                if len(self.his_vector) == self.history_size:
                    self.his_vector.pop(0)
                self.his_vector.append((g_ch, p_ch, 1.0 / gama))
                self.H_diag = gama / g_ch.dot(g_ch)

            al = [0] * self.history_size
            num = len(self.his_vector)
            for i in range(num - 1, -1, -1):  # 后向循环.
                g_ch, p_ch, ro = self.his_vector[i]
                al[i] = p_ch.dot(q) * ro
                q -= al[i] * g_ch

            r = self.H_diag * q
            for i in range(num):  # 前向循环.
                g_ch, p_ch, ro = self.his_vector[i]
                beta_i = g_ch.dot(r) * ro
                r += (al[i] - beta_i) * p_ch

            self.adaptive_lr = self.lr
        self.direction = r
        self.params.data += self.adaptive_lr * self.direction
        self.prev_grad = grad.clone()
        self.optimized_params = self.params


class SGDOptimizer(BaseOptimizer):
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

    def step(self):
        """
        添加正则化项后的梯度计算.
        :return:
        """
        self.step_t += 1
        self.direction = self.params.grad + self.gama * self.params.data
        self.update_params()


class ASGDOptimizer(BaseOptimizer):
    """
    随机梯度下降法：实现与梯度下降相同，不同在于训练时，样本损失不同.
    """

    def __init__(self, params, lr=0.01, t0=100):
        """

        :param params:
        :param lr:
        :param gama: 正则化项系数.
        """
        super().__init__(params=params, lr=lr, with_sgd=True)
        self.optimized_params = torch.zeros_like(params)
        self.t0 = t0
        self.mu = 1.0

    def step(self):
        """
        :return:
        """
        self.step_t += 1
        self.params.data -= self.adaptive_lr * self.params.grad  # update parameter

        # averaging
        if self.step_t - self.t0 > 1:  # t等价于算法中的t+1.  t0+1, t0+2, ... , t. 总个数t-t0.
            self.mu = 1 / (self.step_t - self.t0)
            self.optimized_params = (1 - self.mu) * self.optimized_params + self.mu * self.params
        else:
            self.optimized_params = self.params


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

    def step(self):
        """
        顺序执行每个更新器，并应用更新策略.
        :return:
        """
        self.step_t += 1
        self.update_params()


class FirstMomentumOptimizer(BaseOptimizer):
    """
    动量优化器基类.
    """

    def __init__(self, params, lr=0.01, eps=1e-8, momentum_one=1.0, momentum_two=1.0, with_sgd=False):
        """
        初始化方法.
        :param params: 待优化参数.
        :param lr: 学习率.
        :param eps: 修正分母.
        :param momentum_one: 动量系数1.
        :param momentum_two: 动量系数2.
        :param with_sgd: 是否支持随机梯度下降法.
        """
        super().__init__(params, lr=lr, with_sgd=with_sgd)
        self.momentum_one = momentum_one
        self.momentum_two = momentum_two
        self.eps = eps
        self.m = torch.zeros_like(params)  # 梯度一阶矩.
        self.m_hat = torch.zeros_like(params)  # 修正的梯度一阶矩.

    def update_m(self):
        """
        更新梯度一阶矩.
        :return:
        """
        self.m = self.momentum_one * self.m + self.momentum_two * self.params.grad

        return self.m

    def correct_m(self):
        """
        修正梯度一阶矩.
        :return:
        """
        self.m_hat = self.m / (1 - self.momentum_one ** self.step_t)

        return self.m_hat


class MomentumOptimizer(FirstMomentumOptimizer):
    """
    动量法优化器.
    """

    def __init__(self, params, lr=0.01, momentum=0.9, with_sgd=False, version=0):
        if version in [0, 1, 2, 3]:
            momentum_two = [1, -lr, -1, lr][version]
        else:
            momentum_two = 1
        super().__init__(params=params, lr=lr, momentum_one=momentum, momentum_two=momentum_two, with_sgd=with_sgd)

    def step(self):
        """
        顺序执行每个更新器，并应用更新策略.
        :return:
        """
        self.step_t += 1
        self.update_m()
        self.direction = self.m
        self.adaptive_lr = self.lr / self.momentum_two
        self.update_params()


class NAGOptimizer(FirstMomentumOptimizer):
    """
    梯度加速算法.
    """

    def __init__(self, params, lr=1e-3, momentum=0.9, version=0):

        if version in [0, 1, 2, 3]:
            momentum_two = [1, -1, -lr, lr][version]
        else:
            momentum_two = 1
        super().__init__(params=params, lr=lr, momentum_one=momentum, momentum_two=momentum_two)
        self.adaptive_lr = self.lr / self.momentum_two

    def step(self, closure=None):
        old_params_data = self.params.data.clone()  # 存储旧data的值.
        self.params.data = self.params.data - self.lr / self.momentum_two * self.momentum_one * self.m  # 下一步参数的值.
        # 计算下一步参数的梯度.
        closure()
        self.update_m()
        self.params.data = old_params_data
        self.direction = self.m
        self.update_params()


class NesterovOptimizer(FirstMomentumOptimizer):
    """
    动量加速算法.
    """

    def __init__(self, params, lr=1e-3, momentum=0.9):
        super().__init__(params=params, lr=lr, momentum_one=momentum, momentum_two=1)

    def step(self):
        self.step_t += 1
        self.update_m()  # 先更新动量.
        self.direction = self.momentum_one * self.m + self.params.grad  # 迭代方向提前使用下一步动量.
        self.update_params()


class AggMoOptimizer(BaseOptimizer):
    """
    AggMo算法.
    """

    def __init__(self, params, lr=1e-3, momentum_tuple=(0.0, 0.9, 0.99)):
        super().__init__(params, lr)
        self.k = len(momentum_tuple)  # 动量数量.
        self.momentum_s = torch.tensor(momentum_tuple).view(-1, 1)  # 动量系数张量, shape(k,1).
        # +表示形状拼接.
        self.ms = torch.zeros((self.k,) + self.params.shape)  # 动量张量, shape(k,params.shape).

    def step(self):
        self.step_t += 1
        self.ms = self.momentum_s * self.ms + self.params.grad  # 更新动量张量.
        agg_m = torch.mean(self.ms, dim=0)  # 按照动量系数维度聚合动量.
        self.direction = agg_m
        self.update_params()


class SecMomentOptimizer(BaseOptimizer):
    """
    动量优化器基类.
    """

    def __init__(self, params, lr=0.01, eps=1e-8, decay_factor_one=1.0, decay_factor_two=1.0, with_sgd=False):
        """
        初始化方法.
        :param params: 待优化参数.
        :param lr: 学习率.
        :param eps: 修正分母.
        :param decay_factor_one: 动量系数1.
        :param decay_factor_two: 动量系数2.
        :param with_sgd: 是否支持随机梯度下降法.
        """
        super().__init__(params, lr=lr, with_sgd=with_sgd)
        self.decay_factor_one = decay_factor_one
        self.decay_factor_two = decay_factor_two
        self.eps = eps
        self.s = torch.zeros_like(params)  # 梯度二阶矩.
        self.s_hat = torch.zeros_like(params)  # 修正的梯度二阶矩.

    def update_s(self):
        """
        更新梯度二阶矩.
        :return:
        """
        self.s = self.decay_factor_one * self.s + self.decay_factor_two * self.params.grad ** 2

        return self.s

    def correct_s(self):
        """
        修正梯度二阶矩.
        :return:
        """
        self.s_hat = self.s / (1 - self.decay_factor_one ** self.step_t)

        return self.s_hat


class AdaGradOptimizer(SecMomentOptimizer):
    """
    AdaGrad.
    """

    def __init__(self, params, lr=0.01, eps: float = 1e-10):
        super().__init__(params, lr=lr, eps=eps, decay_factor_one=1.0, decay_factor_two=1.0)

    def step(self):
        self.step_t += 1
        self.update_s()
        self.direction = self.params.grad / torch.sqrt(self.s + self.eps)
        self.update_params()


class RMSPropOptimizer(SecMomentOptimizer):
    """
    RMSProp.
    """

    def __init__(self, params, lr=1e-2, decay_factor=0.99, eps=1e-8):
        super().__init__(params, lr, eps=eps, decay_factor_one=decay_factor, decay_factor_two=1 - decay_factor)

    def step(self):
        self.step_t += 1
        self.update_s()
        self.direction = self.params.grad / torch.sqrt(self.s + self.eps)
        self.update_params()


class AdaDeltaOptimizer(SecMomentOptimizer):
    def __init__(self, params, lr=1.0, eps=1e-6, decay_factor=0.9):
        super().__init__(params, lr=lr, eps=eps, decay_factor_one=decay_factor, decay_factor_two=1 - decay_factor)
        self.delta_square = torch.zeros_like(params)

    def step(self):
        self.step_t += 1
        self.update_s()
        delta = torch.sqrt(self.delta_square + self.eps) / torch.sqrt(self.s + self.eps) * self.params.grad
        self.delta_square = self.decay_factor_one * self.delta_square + self.decay_factor_two * delta * delta
        self.direction = delta
        self.update_params()


class AdamOptimizer(FirstMomentumOptimizer, SecMomentOptimizer):
    """
    Adam.
    """

    def __init__(self, params, lr=0.01, eps=1e-8, momentum=0.9, decay_factor=0.999, with_sgd=False):
        FirstMomentumOptimizer.__init__(self, params=params, lr=lr, eps=eps,
                                        momentum_one=momentum, momentum_two=1 - momentum, with_sgd=with_sgd)
        SecMomentOptimizer.__init__(self, params=params, lr=lr, eps=eps,
                                    decay_factor_one=decay_factor, decay_factor_two=1 - decay_factor,
                                    with_sgd=with_sgd)

    def step(self):
        self.step_t += 1
        self.update_m()
        self.update_s()
        self.correct_m()
        self.correct_s()
        self.direction = self.get_m_direction()
        self.update_params()

    def get_m_direction(self):
        """
        获取动量方向.
        :return:
        """
        return self.m_hat / (torch.sqrt(self.s_hat) + self.eps)


class AdamMaxOptimizer(AdamOptimizer):

    def step(self):
        self.step_t += 1
        self.update_m()
        self.correct_m()
        norm = torch.abs(self.params.grad) + self.eps
        self.s = torch.max(self.decay_factor_one * self.s, norm)
        self.direction = self.m_hat / self.s
        self.update_params()


class AdamWOptimizer(AdamOptimizer):
    def __init__(self, params, lr=2e-3, eps: float = 1e-8, momentum=0.9, decay_factor=0.999,
                 weight_decay=1e-2, alpha=1.0):
        super().__init__(params=params, lr=lr, eps=eps, momentum=momentum, decay_factor=decay_factor)
        self.weight_decay = weight_decay
        self.alpha = alpha

    def step(self):
        self.step_t += 1
        self.update_m()
        self.update_s()
        self.correct_m()
        self.correct_s()
        self.direction = self.alpha * self.get_m_direction() + self.weight_decay * self.params.data
        self.update_params()


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
        self.m = torch.zeros_like(self.params)
        self.weight_decay = weight_decay
        self.momentum_one = momentum
        self.momentum_two = alpha * lr

    def step(self):
        self.step_t += 1
        self.m = self.momentum_one * self.m + self.momentum_two * self.params.grad
        self.params.data -= self.m + self.lr * self.gama * self.params.data
        self.optimized_params = self.params


class AMSGradOptimizer(AdamOptimizer):
    def __init__(self, params, lr=0.01, eps=1e-8, momentum=0.9, decay_factor=0.999):
        super().__init__(params, lr=lr, eps=eps, momentum=momentum, decay_factor=decay_factor)
        self.max_s = torch.zeros_like(self.params)

    def step(self):
        self.step_t += 1
        self.update_m()
        self.update_s()
        self.correct_m()
        self.max_s = torch.max(self.max_s, self.s)
        correct_max_s = self.max_s / (1 - self.decay_factor_one ** self.step_t)  # 修正.
        self.direction = self.m_hat / (torch.sqrt(correct_max_s) + self.eps)
        self.update_params()


class AdafactorOptimizer(AdamOptimizer):

    def __init__(self, params, lr=0.01, eps=1e-30, eps2=1e-3, momentum=0.0, decay_factor_exp=-0.8, d=1.0):
        super().__init__(params, lr=lr, eps=eps, momentum=momentum, decay_factor=decay_factor_exp)
        self.eps2 = eps2
        self.d = d
        self.decay_factor_exp = decay_factor_exp
        if self.params.dim() > 1:
            self.row = torch.zeros(size=(self.params.size(0), 1))
            self.col = torch.zeros(size=(1, self.params.size(1)))

    def step(self):
        self.step_t += 1
        self.update_m()
        self.decay_factor_two = self.step_t ** self.decay_factor_exp
        self.decay_factor_one = 1 - self.decay_factor_two

        if self.params.grad.dim() > 1:  # 矩阵形式.
            self.grad_loss_matrix()
        else:  # 向量形式.
            self.update_s()
        u = self.m / (torch.sqrt(self.s) + self.eps)
        rms_u = u.norm(2).item() / u.numel() ** 0.5
        self.direction = u / max(1.0, rms_u / self.d)

        rms_params = self.params.norm(2).item() / self.params.numel() ** 0.5
        rho_t = min(self.lr, 1 / self.step_t ** 0.5)
        self.adaptive_lr = max(self.eps2, rms_params) * rho_t

        self.update_params()

    def grad_loss_matrix(self):

        r = self.params.grad.size(0)
        c = self.params.grad.size(-1)
        one_row = torch.ones((1, r))
        one_col = torch.ones((c, 1))

        grad = self.params.grad
        self.row = self.decay_factor_one * self.row + self.decay_factor_two * torch.matmul(grad * grad, one_col)
        self.col = self.decay_factor_one * self.col + self.decay_factor_two * torch.matmul(one_row, grad * grad)
        self.s = torch.matmul(self.row, self.col / torch.matmul(one_row, self.row))


class AdaBoundOptimizer(AdamOptimizer):
    """
    Adam.
    """

    def __init__(self, params, lr=0.01, final_lr: float = 0.1, eps=1e-8, momentum=0.9, decay_factor=0.999,
                 amsbound: bool = False):
        super().__init__(params, lr=lr, eps=eps, momentum=momentum, decay_factor=decay_factor)
        self.final_lr = final_lr
        self.amsbound = amsbound
        self.s_max = torch.zeros_like(self.params)

    def step(self):
        self.step_t += 1
        self.update_m()
        self.update_s()
        self.correct_m()

        tmp = (1 - self.decay_factor_one) * self.step_t
        lower_bound = self.final_lr * (1 - 1.0 / (tmp + 1))
        upper_bound = self.final_lr * (1 + 1.0 / tmp)

        if self.amsbound:
            self.s_max = torch.max(self.s_max, self.s)
            correct_max_s = self.s_max / (1 - self.decay_factor_one ** self.step_t)  # 修正.
            step_size = torch.clamp(self.lr / (torch.sqrt(correct_max_s) + self.eps), lower_bound, upper_bound)
        else:
            self.correct_s()
            step_size = torch.clamp(self.lr / (torch.sqrt(self.s_hat) + self.eps), lower_bound, upper_bound)
        self.params.data -= step_size * self.m_hat
        self.optimized_params = self.params


class AMSBoundOptimizer(AdaBoundOptimizer):
    """
    Adam.
    """

    def __init__(self, params, lr=0.01, final_lr: float = 0.1, eps=1e-8, momentum=0.9, decay_factor=0.999):
        super().__init__(params=params, lr=lr, final_lr=final_lr, eps=eps,
                         momentum=momentum, decay_factor=decay_factor, amsbound=True)


class RAdamOptimizer(AdamOptimizer):
    def __init__(self, params, lr=1e-3, eps=1e-8, momentum=0.9, decay_factor=0.999):
        super().__init__(params, lr=lr, eps=eps, momentum=momentum, decay_factor=decay_factor)
        self.rho_inf = 2 / (1 - decay_factor) - 1

    def step(self):
        self.step_t += 1
        self.update_m()
        self.update_s()
        self.correct_m()
        a = self.decay_factor_one ** self.step_t
        rho_t = self.rho_inf - 2 * self.step_t * a / (1 - a)
        if rho_t > 5:  # 延迟修正.
            self.correct_s()
            b = (rho_t - 4) * (rho_t - 2) * self.rho_inf
            c = (self.rho_inf - 4) * (self.rho_inf - 2) * rho_t
            r_t = math.sqrt(b / c)
            self.direction = r_t * self.get_m_direction()
        else:
            self.direction = self.m_hat
        self.update_params()


class NadamOptimizer(AdamOptimizer):
    def __init__(self, params, lr=1e-3, eps=1e-8, momentum=0.9, decay_factor=0.999, momentum_decay=4e-3):
        super().__init__(params, lr=lr, eps=eps, momentum=momentum, decay_factor=decay_factor)
        self.momentum_decay = momentum_decay
        self.mu_product = 1.0

    def step(self):
        self.step_t += 1
        self.update_m()
        self.update_s()

        mu = self.momentum_one * (1.0 - 0.5 * (0.96 ** (self.step_t * self.momentum_decay)))
        mu_next = self.momentum_one * (1.0 - 0.5 * (0.96 ** ((self.step_t + 1) * self.momentum_decay)))
        self.mu_product *= mu
        self.m_hat = self.m / (1 - self.mu_product * mu_next)

        self.correct_s()
        grad_hat = self.params.grad / (1 - self.mu_product)

        m_bar = mu_next * self.m_hat + (1 - mu) * grad_hat
        self.direction = m_bar / (torch.sqrt(self.s_hat) + self.eps)
        self.update_params()


class QHAdamOptimizer(AdamOptimizer):
    def __init__(self, params, lr=1e-3, eps=1e-8, momentum=0.9, decay_factor=0.999, v1=1.0, v2=1.0):
        super().__init__(params, lr=lr, eps=eps, momentum=momentum, decay_factor=decay_factor)
        self.v1 = v1
        self.v2 = v2

    def step(self):
        self.step_t += 1
        self.update_m()
        self.update_s()
        self.correct_m()
        self.correct_s()
        grad = self.params.grad
        numerator = self.v1 * self.m_hat + (1 - self.v1) * self.params.grad
        denominator = self.v2 * self.s_hat + (1 - self.v2) * grad * grad
        self.direction = numerator / (torch.sqrt(denominator) + self.eps)
        self.update_params()


class MADGRADOptimizer(AdamOptimizer):
    def __init__(self, params, lr=0.01, momentum=0.9):
        super().__init__(params=params, lr=lr, momentum=momentum)
        self.init_data = self.params.data.clone()

    def step(self):
        self.step_t += 1
        grad = self.params.grad
        alpha = self.lr * math.sqrt(self.step_t)
        self.m += alpha * grad
        self.s += alpha * grad * grad
        z = self.init_data - self.m / (torch.pow(self.s, 1 / 3) + self.eps)
        self.params.data = self.momentum_one * self.params.data + (1 - self.momentum_one) * z
        self.optimized_params = self.params


class SWATSOptimizer(AdamOptimizer):
    def __init__(self, params, lr=1e-3, eps=1e-3, momentum=0.9, decay_factor=0.999, e=1e-6, sim_e=0):
        """
        初始化方法.
        :param params: 待优化参数.
        :param lr:
        :param eps:
        :param momentum:
        :param decay_factor:
        :param e:
        :param sim_e:
        """
        super().__init__(params, lr=lr, eps=eps, momentum=momentum, decay_factor=decay_factor)
        self.sgd = 'SGD'
        self.adam = 'Adam'
        self.phase = self.adam
        self.v = torch.zeros_like(self.params)
        self.delta = 0
        self.e = e
        self.sim_e = sim_e

    def step(self):
        self.step_t += 1
        if self.phase == self.adam:  # Adam阶段.
            self.update_m()  # 更新梯度一阶矩.
            self.update_s()  # 更新梯度二阶矩.
            self.correct_m()  # 修正梯度一阶矩.
            self.correct_s()  # 修正梯度二阶矩.
            p = -self.lr * self.get_m_direction()
            self.params.data += p  # 参数更新.
            sim = torch.sum(p * self.params.grad).item()
            if abs(sim) != self.sim_e:
                gama = - torch.sum(p * p) / sim
                self.delta = self.decay_factor_one * self.delta + self.decay_factor_two * gama
                yita = self.delta / (1 - self.decay_factor_one ** self.step_t)  # SGD学习率.
                if self.step_t > 1 and abs(yita - gama) < self.e:  # 阶段切换.
                    self.phase = self.sgd
                    print(f'{self.adam}切换到{self.sgd}: t={self.step_t}')
                    self.adaptive_lr = (1 - self.momentum_one) * yita  # 设置SGD的自适应学习率.
        else:  # SGD阶段.
            self.v = self.momentum_one * self.v + self.params.grad  # 更新SGD的梯度一阶矩.
            self.params.data -= self.adaptive_lr * self.v  # 参数更新.
        self.optimized_params = self.params


class YogiOptimizer(AdamOptimizer):

    def step(self):
        self.step_t += 1
        self.update_m()
        self.correct_m()
        grad_square = self.params.grad * self.params.grad
        self.s += self.decay_factor_two * torch.sign(grad_square - self.s) * grad_square
        self.correct_s()
        self.direction = self.m_hat / (torch.sqrt(self.s_hat) + self.eps)
        self.update_params()


class ShampooOptimizer(BaseOptimizer):

    def __init__(self, params, lr=0.01, eps=1e-4, version=0):
        super().__init__(params, lr=lr, eps=eps)
        self.version = version
        self.order = self.params.dim()
        self.left = self.eps * torch.eye(self.params.size(0))
        self.right = self.eps * torch.eye(self.params.size(1))
        self.is_shampoo = True

    def step(self):
        self.step_t += 1
        if self.version == 0:
            self.user_version()  # 笔者实现.
        else:
            self.torch_optimizer_version()  # torch_optimizer.Shampoo实现
        self.update_params()

    def user_version(self):
        """
        笔者实现.
        :return:
        """
        grad = self.params.grad
        grad_t = grad.T
        self.left += grad @ grad_t
        self.right += grad_t @ grad
        left_inv = self.matrix_power(self.left, -1 / (2 * self.order))
        right_inv = self.matrix_power(self.right, -1 / (2 * self.order))
        self.direction = left_inv @ grad @ right_inv

    def torch_optimizer_version(self):
        """
        torch_optimizer.Shampoo实现.
        :return:
        """
        grad = self.params.grad
        grad_t = grad.T
        self.left += grad @ grad_t
        left_inv = self.matrix_power(self.left, -1 / self.order)
        grad = left_inv @ grad
        grad_t = grad.T
        self.right += grad_t @ grad
        right_inv = self.matrix_power(self.right, -1 / self.order)
        self.direction = grad @ right_inv

    def matrix_power(self, matrix: torch.Tensor, power: float) -> torch.Tensor:
        u, s, v = torch.svd(matrix)
        return u @ s.pow_(power).diag() @ v.t()


class LookaheadOptimizer(BaseOptimizer):
    """
    Lookahead进行快权重更新.
    """

    def __init__(self, optimizer: BaseOptimizer, p: int = 5, alpha: float = 0.5, version=0):
        super().__init__(params=optimizer.params, lr=optimizer.lr)
        self.optimizer = optimizer  # 内部优化器.
        self.p = p
        self.alpha = alpha
        self.slow_points = []
        self.slow_param = None
        self.n = 0
        self.version = version

    def step(self):
        self.step_t += 1
        loss = self.optimizer.step()  # 快权重更新.

        if self.version == 0:
            if self.n == 0:
                self._update()  # 慢参数更新.
            self.n += 1
            self.n %= self.p
        else:
            if self.slow_param is None:
                self.slow_param = self.optimizer.params.data.clone()
                self.slow_points.append(self.slow_param.clone())
            if self.step_t % self.p == 0:  # 慢权重更新.
                self.slow_param += self.alpha * (self.optimizer.params.data - self.slow_param)
                self.slow_points.append(self.slow_param.clone())  # 记录慢参数.
                self.optimizer.params.data = self.slow_param  # 使用慢权重更新快权重.
        self.optimized_params = self.params

        return loss

    def _update(self) -> None:
        if self.slow_param is None:  # 慢权重初始化.
            self.slow_param = torch.clone(self.params.data).detach()
            self.slow_points.append(self.slow_param.clone())
        fast = self.params
        fast.data.mul_(self.alpha).add_(self.slow_param, alpha=1.0 - self.alpha)
        self.slow_param.data.copy_(fast)
        self.slow_points.append(self.slow_param.clone())  # 记录慢参数.


class LARSOptimizer(MomentumOptimizer):
    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=1e-2, lars_c=0.01):
        super().__init__(params=params, lr=lr, momentum=momentum, version=0)
        self.weight_decay = weight_decay
        self.lars_c = lars_c

    def step(self):
        self.step_t += 1
        grad = self.params.grad
        data = self.params.data
        data_norm = torch.norm(data)
        grad_norm = torch.norm(grad)
        lars_lr = self.lars_c * data_norm / (grad_norm + self.weight_decay * data_norm + self.eps)
        self.m = self.momentum_one * self.m + lars_lr * (grad + self.weight_decay * data)
        self.direction = self.m
        self.update_params()


class LambOptimizer(AdamOptimizer):
    def __init__(self, params, lr=0.01, momentum=0.9, decay_factor=0.999, weight_decay=1e-2, clamp_value: float = 10):
        super().__init__(params=params, lr=lr, momentum=momentum, decay_factor=decay_factor)
        self.weight_decay = weight_decay
        self.clamp_value = clamp_value

    def step(self):
        self.step_t += 1
        self.update_m()
        self.update_s()
        self.correct_m()
        self.correct_s()
        r = self.get_m_direction() + self.weight_decay * self.params.data
        self.direction = torch.norm(self.params.data).clamp(0, self.clamp_value) * r / torch.norm(r)
        self.update_params()


class AdaBeliefOptimizer(AdamOptimizer):
    def __init__(self, params, lr=0.01, eps=1e-8, momentum=0.9, decay_factor=0.999):
        super().__init__(params, lr=lr, eps=eps, momentum=momentum, decay_factor=decay_factor)

    def step(self):
        self.step_t += 1
        self.update_m()
        self.s = self.decay_factor_one * self.s + self.decay_factor_two * (self.params.grad - self.m) ** 2
        self.correct_m()
        self.correct_s()
        self.direction = self.get_m_direction()
        self.update_params()


class LionOptimizer(AdamOptimizer):
    def __init__(self, params, lr=0.01, eps=1e-8, momentum=0.9, decay_factor=0.99, weight_decay=0.0):
        super().__init__(params, lr=lr, eps=eps, momentum=momentum, decay_factor=decay_factor)
        self.weight_decay = weight_decay

    def step(self):
        self.step_t += 1
        update = self.momentum_one * self.m + self.momentum_two * self.params.grad
        self.m = self.decay_factor_one * self.m + self.decay_factor_two * self.params.grad
        self.params.data = (1.0 - self.weight_decay * self.lr) * self.params.data - self.lr * update.sign_()
        self.optimized_params = self.params


class SophiaHOptimizer(AdamOptimizer):
    def __init__(self, params, lr=6e-2, momentum=0.96, decay_factor=0.99, weight_decay=0.0, eps=1e-12,
                 delta: float = 1e-2, period=10, num_samples=1, distribution='gaussian'):
        """

        :param params: 参数.
        :param lr: 学习率.
        :param momentum: 动量参数.
        :param decay_factor: hessian动量参数.
        :param weight_decay: 权重衰减系数.
        :param delta: 裁剪阈值.
        :param period: 迭代周期.
        """
        super().__init__(params=params, lr=lr, momentum=momentum, eps=eps, decay_factor=decay_factor)
        self.h, self.diag_hessian = torch.zeros_like(self.params), torch.zeros_like(self.params)
        self.weight_decay = weight_decay
        self.delta = delta
        self.period, self.num_samples = period, num_samples
        self.distribution = distribution

    def step(self):
        self.step_t += 1
        self.update_m()
        if self.step_t % self.period == 1:
            self.diag_hessian.zero_()  # 将hessian对角元素置为0.
            diag_hessian = self.compute_hutchinson_hessian()
            self.h = self.decay_factor_one * self.h + self.decay_factor_two * diag_hessian
        u = self.m / torch.clip(self.delta * self.h, min=self.eps)
        self.direction = torch.clamp(u, -1, 1)
        self.update_params()

    def compute_hutchinson_hessian(self):
        for i in range(self.num_samples):
            ui = torch.randint_like(self.params, 0, 1) * 2.0 - 1.0 \
                if self.distribution == 'rademacher' else torch.randn_like(self.params)
            hui = torch.autograd.grad(self.params.grad, self.params, grad_outputs=ui,
                                      retain_graph=i < self.num_samples - 1)[0]  # h * ui.
            self.diag_hessian += hui * ui
        return self.diag_hessian / self.num_samples


class RpropOptimizer(BaseOptimizer):
    def __init__(self, params, lr=0.1, eta_plus=1.2, eta_minus=0.5, delta_min=1e-6, delta_max=50):
        super().__init__(params, lr=lr)
        self.eta_plus = eta_plus
        self.eta_minus = eta_minus
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.pre_grad = torch.zeros_like(self.params)
        self.delta = torch.full_like(self.params, self.lr)
        self.adaptive_lr = 1.0

    def step(self):
        self.step_t += 1
        grad = self.params.grad
        sign = grad.mul(self.pre_grad).sign()  # 计算当前时刻梯度与前一时刻梯度乘积符号.
        self.pre_grad.copy_(grad)  # 当前梯度复制给前一时刻梯度.
        # 根据梯度符号设置步长因子.
        sign[sign.gt(0)] = self.eta_plus
        sign[sign.lt(0)] = self.eta_minus
        sign[sign.eq(0)] = 1.0
        grad[sign.eq(self.eta_minus)] = 0  # 如果梯度方向改变，梯度置为0.  iRprop+.

        self.delta.mul_(sign).clamp_(self.delta_min, self.delta_max)
        self.direction = grad.sign() * self.delta
        self.update_params()
