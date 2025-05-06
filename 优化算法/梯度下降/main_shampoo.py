import torch
from common import VirtualModel
from common import SquareFunction
from optimizer import ShampooOptimizer, SystemOptimizer
from common import OptimizerComparator
from torch_optimizer import Shampoo
import matplotlib.pyplot as plt
from common import plot_contour, plot_points

if __name__ == '__main__':
    # 向量形式.
    lr = 5
    configs = [{'type': 'system', 'class': Shampoo, 'kwargs': {'lr': lr}},
               {'type': 'custom', 'class': ShampooOptimizer, 'kwargs': {'lr': lr}}]
    comparator = OptimizerComparator(configs=configs)
    comparator.compare()

    # 矩阵形式.
    lr = 5
    criterion = SquareFunction(torch.tensor([[1.0],
                                             [2.0]]))
    plot_contour(criterion, r=15, n=400)  # 绘制等高线.
    for i in range(2):
        if i == 0:
            params = torch.tensor([[10.0],
                                   [10.0]], requires_grad=True)
            optimizer = SystemOptimizer(Shampoo(params=[params], lr=lr))
        else:
            params = torch.tensor([[-10.0],
                                   [10.0]], requires_grad=True)
            optimizer = ShampooOptimizer(params=params, lr=lr)
        model = VirtualModel(
            criterion=criterion,
            optimizer=optimizer
        )
        model.fit()
        plot_points(model.points, optimizer)  # 绘制参数迭代路径.
    plt.show()
