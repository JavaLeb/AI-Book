from torch_optimizer import Lookahead
from common import OptimizerComparator
from optimizer import LookaheadOptimizer, GDOptimizer, SystemOptimizer
import torch
from common import SquareFunction, plot_contour, VirtualModel, plot_points, plot_curve, generate_data, MSELoss, \
    plot_contour_with_sample
from common import LinearRegression
from matplotlib import pyplot as plt

if __name__ == '__main__':
    lr = 0.1
    # 内置lookahead.
    initial_param1 = torch.tensor([-10.0, 10.0], requires_grad=True)
    sgd = torch.optim.SGD(params=[initial_param1], lr=lr)  # pytorch内置sgd.
    lookahead = Lookahead(optimizer=sgd, k=5, alpha=0.1)  # 内置lookahead.
    system_optimizer = SystemOptimizer(lookahead)

    # 自定义实现lookahead, version=1.
    initial_param2 = torch.tensor([10.0, 10.0], requires_grad=True)
    lookahead_optimizer_0 = LookaheadOptimizer(optimizer=GDOptimizer(params=initial_param2, lr=lr), p=5,
                                               alpha=0.1, version=0)  # 自定义lookahead.

    # 自定义实现lookahead, version=0.
    initial_param3 = torch.tensor([-10.0, -10.0], requires_grad=True)
    lookahead_optimizer_1 = LookaheadOptimizer(optimizer=GDOptimizer(params=initial_param3, lr=lr), p=5,
                                               alpha=0.1, version=1)  # 自定义lookahead.

    # gd.
    initial_param4 = torch.tensor([10.0, -10.0], requires_grad=True)
    gd = GDOptimizer(params=initial_param4, lr=lr)  # 自定义实现gd.

    optimizers = [system_optimizer, lookahead_optimizer_0, lookahead_optimizer_1, gd]

    # 多元函数.
    criterion = SquareFunction(torch.tensor([1.0, 2.0]))
    plot_contour(criterion, r=15, n=400)  # 绘制等高线.
    models = []
    for optimizer in optimizers:
        model = VirtualModel(
            criterion=criterion,
            optimizer=optimizer
        )
        model.fit()
        models.append(model)
        plot_points(model.points, optimizer)  # 绘制参数迭代路径.
    plot_points(lookahead_optimizer_0.slow_points, lookahead_optimizer_0, annotate=False)
    plot_points(lookahead_optimizer_1.slow_points, lookahead_optimizer_1, annotate=False)
    plt.figure(figsize=(10, 8))
    for model in models:
        plot_curve(model)
    plt.show()

    # # 线性回归.
    # X, y = generate_data()
    # criterion = MSELoss()
    # plot_contour_with_sample(criterion, X, y, r=11, n=400)  # 绘制等高线.
    # for optimizer in optimizers:
    #     model = LinearRegression(
    #         in_features=1,
    #         out_features=1,
    #         criterion=criterion,
    #         optimizer=optimizer
    #     )
    #     model.fit(X, y, epochs=2000, batch_size=5, shuffled=True)
    #     plot_points(model.points, optimizer, annotate=False)  # 绘制参数迭代路径.
    # plt.show()
