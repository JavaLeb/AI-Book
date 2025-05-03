import torch
import torch_optimizer

from common import VirtualModel
from common import SquareFunction
from optimizer import YogiOptimizer

import matplotlib.pyplot as plt
from common import plot_contour, plot_points
from common import OptimizerComparator

from torch.optim import Adadelta, Adagrad

if __name__ == '__main__':
    lr = 0.1
    configs = [{'type': 'system', 'class': torch_optimizer.Yogi, 'kwargs': {'lr': lr}},
               {'type': 'custom', 'class': YogiOptimizer, 'kwargs': {'lr': lr}}]
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
