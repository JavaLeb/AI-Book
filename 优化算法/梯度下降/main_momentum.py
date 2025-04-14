import torch
from common import VirtualModel
from common import SquareFunction
from optimizer import MomentumOptimizer

import matplotlib.pyplot as plt
from common import plot_contour, plot_points

if __name__ == '__main__':
    initial_param = torch.tensor([10.0, 10.0], requires_grad=True)
    loss = SquareFunction(torch.tensor([1.0, 2.0]))
    model = VirtualModel(
        optimizer=MomentumOptimizer(loss=loss, params=initial_param, lr=0.01)
    )
    model.fit()

    # 演示不同学习率
    initial_params = [torch.tensor([10.0, 10.0], requires_grad=True),
                      torch.tensor([-10.0, 10.0], requires_grad=True),
                      torch.tensor([-10.0, -10.0], requires_grad=True)]
    learning_rates = [0.1]
    plt.figure(figsize=(10, 8))
    loss = SquareFunction(torch.tensor([1.0, 2.0]))
    for i in range(len(learning_rates)):
        lr = learning_rates[i]
        model = VirtualModel(
            optimizer=MomentumOptimizer(loss=loss, params=initial_params[i], lr=lr)
        )
        model.fit()
        plot_points(model.points, lr)  # 绘制参数迭代路径.
    plot_contour(loss, r=15, n=400)  # 绘制等高线.
    plt.show()
