import torch
from common import VirtualModel
from common import SquareFunction
from optimizer import AdaDeltaOptimizer
from torch_optimizer import LARS
import matplotlib.pyplot as plt
from common import plot_contour, plot_points
from common import OptimizerComparator
from optimizer import LARSOptimizer

from torch.optim import Adadelta, Adagrad

if __name__ == '__main__':
    lr = 0.1
    kwargs1 = {'lr': lr, 'momentum': 0.9, 'weight_decay': 0.01, 'trust_coefficient': 0.01}
    kwargs2 = {'lr': lr, 'momentum': 0.9, 'weight_decay': 0.01, 'lars_c': 0.01}
    configs = [{'type': 'system', 'class': LARS, 'kwargs': kwargs1},
               {'type': 'custom', 'class': LARSOptimizer, 'kwargs': kwargs2}]
    comparator = OptimizerComparator(configs=configs)
    comparator.compare()

    # # 演示不同学习率
    # initial_params = [torch.tensor([10.0, 10.0], requires_grad=True),
    #                   torch.tensor([-10.0, 10.0], requires_grad=True),
    #                   torch.tensor([-10.0, -10.0], requires_grad=True)]
    # learning_rates = [5]
    # plt.figure(figsize=(10, 8))
    # loss = SquareFunction(torch.tensor([1.0, 2.0]))
    # for i in range(len(learning_rates)):
    #     lr = learning_rates[i]
    #     model = VirtualModel(
    #         criterion=loss,
    #         optimizer=AdaGrad(params=initial_params[i], lr=lr)
    #     )
    #     model.fit()
    #     plot_points(model.points, lr)  # 绘制参数迭代路径.
    # plot_contour(model.criterion, r=15, n=400)  # 绘制等高线.
    # plt.show()
