import torch
from common import VirtualModel
from common import SquareFunction
from optimizer import AdaGrad, AdaDelta, Adam

import matplotlib.pyplot as plt
from common import plot_contour, plot_points

if __name__ == '__main__':
    initial_param = torch.tensor([10.0, 10.0], requires_grad=True)
    loss = SquareFunction(torch.tensor([1.0, 2.0]))
    # optimizer=AdaGrad(params=initial_param, lr=10)
    optimizer = Adam(params=initial_param, lr=0.004)
    model = VirtualModel(
        criterion=loss,
        optimizer=optimizer
    )
    model.fit()
    plt.figure(figsize=(10, 8))
    plot_points(model.points, optimizer.lr)  # 绘制参数迭代路径.
    plot_contour(model.criterion, r=15, n=400)  # 绘制等高线.
    plt.show()

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
