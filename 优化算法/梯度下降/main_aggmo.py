import torch
from common import VirtualModel
from common import SquareFunction
from optimizer import AggMoOptimizer
from torch_optimizer import AggMo

import matplotlib.pyplot as plt
from common import plot_contour, plot_points
from common import OptimizerComparator

if __name__ == '__main__':
    kwargs = {'lr': 0.5}
    configs = [{'type': 'system', 'class': AggMo, 'kwargs': kwargs},
               {'type': 'custom', 'class': AggMoOptimizer, 'kwargs': kwargs}]
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
