import torch
from common import VirtualModel, SquareFunction, plot_points, plot_contour
from optimizer import NewtonOptimizer
import matplotlib.pyplot as plt
from common import OptimizerComparator

if __name__ == "__main__":
    kwargs = {'lr': 0.1}
    configs = [{'type': 'system', 'class': torch.optim.LBFGS, 'kwargs': kwargs},
               {'type': 'custom', 'class': NewtonOptimizer, 'kwargs': kwargs}]
    comparator = OptimizerComparator(configs=configs)
    comparator.compare()


    # initial_params = torch.tensor([10.0, 10.0], requires_grad=True)
    # loss = SquareFunction(torch.tensor([1.0, 2.0]))
    # lr = 1
    # optimizer = NewtonOptimizer(params=initial_params, lr=lr)
    # plot_contour(loss, r=15, n=400)  # 绘制等高线.
    # model = VirtualModel(
    #     criterion=loss,
    #     optimizer=optimizer
    # )
    # model.fit()
    # plot_points(model.points, optimizer)  # 绘制参数迭代路径.
    #
    # plt.show()
