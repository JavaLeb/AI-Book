import torch
from common import VirtualModel
from common import SquareFunction
from optimizer import AdaBoundOptimizer,AMSBoundOptimizer
from common import OptimizerComparator
from torch_optimizer import AdaBound

import matplotlib.pyplot as plt
from common import plot_contour, plot_points, plot_curve

if __name__ == '__main__':

    # AdaBound.
    kwargs = {'lr': 5}
    configs = [{'type': 'system', 'class': AdaBound, 'kwargs': kwargs},
               {'type': 'custom', 'class': AdaBoundOptimizer, 'kwargs': kwargs}]
    comparator = OptimizerComparator(configs=configs)
    comparator.compare()

    # AMSBound.
    kwargs1 = {'lr': 5, 'amsbound': True}
    kwargs2 = {'lr': 5}
    configs = [{'type': 'system', 'class': AdaBound, 'kwargs': kwargs1},
               {'type': 'custom', 'class': AMSBoundOptimizer, 'kwargs': kwargs2}]
    comparator = OptimizerComparator(configs=configs)
    comparator.compare()

    # # 演示不同学习率
    # initial_params = [torch.tensor([10.0, 10.0], requires_grad=True),
    #                   torch.tensor([-10.0, 10.0], requires_grad=True),
    #                   torch.tensor([-10.0, -10.0], requires_grad=True)]
    # learning_rates = [0.5]
    # plt.figure(figsize=(10, 8))
    # loss = SquareFunction(torch.tensor([1.0, 2.0]))
    # for i in range(len(learning_rates)):
    #     lr = learning_rates[i]
    #     model = VirtualModel(
    #         optimizer=AdamOptimizer(loss=loss, params=initial_params[i], lr=lr)
    #     )
    #     model.fit()
    #     plot_points(model.points, lr)  # 绘制参数迭代路径.
    # plot_contour(loss, r=15, n=400)  # 绘制等高线.
    # plt.show()
