import torch
from common import VirtualModel, LinearRegression
from common import SquareFunction
from optimizer import GradientDescent

import matplotlib.pyplot as plt
from common import MSELoss, generate_data, plot_contour_with_sample, plot_points
from common import plot_contour

if __name__ == '__main__':
    initial_param = torch.tensor([10.0, 10.0], requires_grad=True)  # 参数初始值.
    loss = SquareFunction(torch.tensor([1.0, 2.0]))  # 定义目标函数.
    optimizer = GradientDescent(params=initial_param, lr=0.1)  # 定义优化器
    model = VirtualModel(loss, optimizer)  # 定义虚拟模型.
    model.fit()  # 模型训练：优化目标函数.

    # 不同学习率演示.
    initial_params = [torch.tensor([10.0, 10.0], requires_grad=True),
                      torch.tensor([-10.0, 10.0], requires_grad=True),
                      torch.tensor([-10.0, -10.0], requires_grad=True)]
    learning_rates = [0.5, 0.45, 0.1]
    plt.figure(figsize=(10, 8))
    loss = SquareFunction(torch.tensor([1.0, 2.0]))
    for i in range(len(learning_rates)):
        lr = learning_rates[i]
        optimizer = GradientDescent(params=initial_params[i], lr=lr)
        model = VirtualModel(loss, optimizer)
        model.fit()
        plot_points(model.points, lr)  # 绘制参数迭代路径.
    plot_contour(loss, r=15, n=400)  # 绘制等高线.
    plt.show()

    # 线性回归.
    initial_param = torch.tensor([10.0, 10.0], requires_grad=True)
    X, y = generate_data()  # X:Tensor(10,1), y:Tensor(10,1)
    loss = MSELoss()
    lr = 0.01
    optimizer = GradientDescent(params=initial_param, lr=lr)
    model = LinearRegression(in_features=1, out_features=1, criterion=loss, optimizer=optimizer)
    model.fit(X, y)

    plt.plot(model.loss_history)  # 学习曲线.
    plt.xlabel("Epochs", fontsize=10)
    plt.ylabel("Loss", fontsize=10)
    plt.title("Training Loss Curve", fontsize=10)

    plot_contour_with_sample(loss, X, y, r=11, n=400)  # 绘制等高线.
    plot_points(model.points, lr, annotate=False)  # 绘制参数迭代路径.
    plt.show()
