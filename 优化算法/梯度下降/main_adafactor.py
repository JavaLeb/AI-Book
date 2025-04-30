import torch
from common import VirtualModel
from common import SquareFunction
from optimizer import AdafactorOptimizer, SystemOptimizer
from common import OptimizerComparator

import matplotlib.pyplot as plt
from common import plot_contour, plot_points, plot_curve

if __name__ == '__main__':
    # 向量形式.
    lr = 0.1
    configs = [{'type': 'system', 'class': torch.optim.Adafactor, 'kwargs': {'lr': lr}},
               {'type': 'custom', 'class': AdafactorOptimizer, 'kwargs': {'lr': lr}}]
    comparator = OptimizerComparator(configs=configs)
    comparator.compare()

    # 矩阵形式.
    lr = 0.1
    criterion = SquareFunction(torch.tensor([[1.0, 2.0, 3.0],
                                             [4.0, 5.0, 6.0]]))
    for i in range(2):
        params = torch.tensor([[10.0, 10.0, 10.0],
                               [10.0, 10.0, 10.0]], requires_grad=True)
        optimizer = SystemOptimizer(torch.optim.Adafactor(params=[params], lr=lr)) if i == 0 else AdafactorOptimizer(
            params=params, lr=lr)
        model = VirtualModel(
            criterion=criterion,
            optimizer=optimizer
        )
        model.fit()

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
