import torch
from common import VirtualModel
from common import SquareFunction
from optimizer import MomentumOptimizer, Momentum2Optimizer, Momentum3Optimizer

import matplotlib.pyplot as plt
from common import plot_contour, plot_points

if __name__ == '__main__':

    loss = SquareFunction(torch.tensor([1.0, 2.0]))
    lr = 0.01
    optimizers = [MomentumOptimizer(params=torch.tensor([10.0, 10.0], requires_grad=True), lr=lr),
                  Momentum2Optimizer(params=torch.tensor([10.0, 10.0], requires_grad=True), lr=lr),
                  Momentum3Optimizer(params=torch.tensor([10.0, 10.0], requires_grad=True), lr=lr)]
    for optimizer in optimizers:
        model = VirtualModel(criterion=loss, optimizer=optimizer)
        model.fit()

    # 演示不同学习率
    initial_params = [torch.tensor([10.0, 10.0], requires_grad=True),
                      torch.tensor([-10.0, 10.0], requires_grad=True),
                      torch.tensor([-10.0, -10.0], requires_grad=True)]
    learning_rates = [0.05,0.01,0.001]
    plt.figure(figsize=(10, 8))
    loss = SquareFunction(torch.tensor([1.0, 2.0]))
    for i in range(len(learning_rates)):
        lr = learning_rates[i]
        model = VirtualModel(
            criterion=loss,
            optimizer=MomentumOptimizer(params=initial_params[i], lr=lr)
        )
        model.fit()
        plot_points(model.points, lr)  # 绘制参数迭代路径.
    plot_contour(loss, r=15, n=400)  # 绘制等高线.
    plt.show()
