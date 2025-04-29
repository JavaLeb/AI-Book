import torch
from common import VirtualModel
from common import SquareFunction, OptimizerComparator
from optimizer import NAGOptimizer,NesterovOptimizer

import matplotlib.pyplot as plt
from common import plot_contour, plot_points

if __name__ == '__main__':
    configs = [{'type': 'system', 'class': torch.optim.SGD, 'kwargs': {'momentum': 0.9, 'lr': 0.1, 'nesterov': True}},
               {'type': 'custom', 'class': NAGOptimizer, 'kwargs': {'momentum': 0.9, 'lr': 0.1}},
               {'type': 'custom', 'class': NesterovOptimizer, 'kwargs': {'momentum': 0.9, 'lr': 0.1}}
               ]
    comparator = OptimizerComparator(configs=configs)
    comparator.compare()

    # # 演示不同学习率
    # initial_params = [torch.tensor([10.0, 10.0], requires_grad=True),
    #                   torch.tensor([-10.0, 10.0], requires_grad=True),
    #                   torch.tensor([-10.0, -10.0], requires_grad=True)]
    # learning_rates = [0.1,0.01,0.001]
    # plt.figure(figsize=(10, 8))
    # loss = SquareFunction(torch.tensor([1.0, 2.0]))
    # for i in range(len(learning_rates)):
    #     lr = learning_rates[i]
    #     model = VirtualModel(
    #         optimizer=NestAcceleratedGradient(loss=loss, params=initial_params[i], lr=lr)
    #     )
    #     model.fit()
    #     plot_points(model.points, lr)  # 绘制参数迭代路径.
    # plot_contour(loss, r=15, n=400)  # 绘制等高线.
    # plt.show()
