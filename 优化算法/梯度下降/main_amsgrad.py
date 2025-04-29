import torch
from common import VirtualModel
from common import SquareFunction
from optimizer import AMSGradOptimizer
from common import OptimizerComparator

import matplotlib.pyplot as plt
from common import plot_contour, plot_points, plot_curve

if __name__ == '__main__':
    lr = 0.9
    kwargs1 = {'lr': lr, 'amsgrad': True}
    kwargs2 = {'lr': lr, 'momentum': 0.9, 'delay_factor': 0.999}
    configs = [{'type': 'system', 'class': torch.optim.Adam, 'kwargs': kwargs1},
               {'type': 'custom', 'class': AMSGradOptimizer, 'kwargs': kwargs2}]
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
