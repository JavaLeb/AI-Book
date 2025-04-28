import torch
from common import VirtualModel, LinearRegression
from common import SquareFunction, OptimizerComparator
from optimizer import MomentumOptimizer

import matplotlib.pyplot as plt
from common import MSELoss, generate_data, plot_contour_with_sample, plot_points, plot_contour

if __name__ == '__main__':
    kwargs = {'momentum': 0.9, 'lr': 0.01}
    configs = [{'type': 'system', 'class': torch.optim.SGD, 'kwargs': kwargs},
               {'type': 'custom', 'class': MomentumOptimizer, 'kwargs': kwargs}]
    comparator = OptimizerComparator(configs=configs)
    comparator.compare()

    # loss = SquareFunction(torch.tensor([1.0, 2.0]))
    # lr = 0.01
    # optimizers = []
    # for version in [0, 1, 2]:
    #     optimizer = MomentumOptimizer(params=torch.tensor([10.0, 10.0], requires_grad=True), lr=lr, version=version)
    #     model = VirtualModel(criterion=loss, optimizer=optimizer)
    #     model.fit()

    # # 演示不同学习率
    # initial_params = [torch.tensor([10.0, 10.0], requires_grad=True),
    #                   torch.tensor([-10.0, 10.0], requires_grad=True),
    #                   torch.tensor([-10.0, -10.0], requires_grad=True)]
    # learning_rates = [0.05, 0.01, 0.001]
    # plt.figure(figsize=(10, 8))
    # loss = SquareFunction(torch.tensor([1.0, 2.0]))
    # for i in range(len(learning_rates)):
    #     lr = learning_rates[i]
    #     model = VirtualModel(
    #         criterion=loss,
    #         optimizer=MomentumOptimizer(params=initial_params[i], lr=lr)
    #     )
    #     model.fit()
    #     plot_points(model.points, lr)  # 绘制参数迭代路径.
    # plot_contour(loss, r=15, n=400)  # 绘制等高线.
    # plt.show()
    #
    # # 线性回归.
    # initial_param = torch.tensor([10.0, 10.0], requires_grad=True)
    # X, y = generate_data()  # X:Tensor(10,1), y:Tensor(10,1)
    # loss = MSELoss()
    # lr = 0.001
    # optimizer = MomentumOptimizer(params=initial_param, lr=lr, with_sgd=True)
    # model = LinearRegression(in_features=1, out_features=1, criterion=loss, optimizer=optimizer)
    # model.fit(X, y, batch_size=10)
    #
    # plt.plot(model.loss_history)  # 学习曲线.
    # plt.xlabel("Epochs", fontsize=10)
    # plt.ylabel("Loss", fontsize=10)
    # plt.title("Training Loss Curve", fontsize=10)
    #
    # plot_contour_with_sample(loss, X, y, r=11, n=400)  # 绘制等高线.
    # plot_points(model.points, lr, annotate=False)  # 绘制参数迭代路径.
    # plt.show()
