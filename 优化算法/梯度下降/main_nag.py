import torch
from common import VirtualModel, LinearRegression
from common import SquareFunction
from optimizer import NAG

import matplotlib.pyplot as plt
from common import MSELoss, generate_data, plot_contour_with_sample, plot_points, plot_contour

if __name__ == '__main__':

    loss = SquareFunction(torch.tensor([1.0, 2.0]))
    lr = 0.01
    optimizers = []
    for version in [0, 1, 2, 3]:
        optimizer = NAG(params=torch.tensor([10.0, 10.0], requires_grad=True), lr=lr, version=version)
        model = VirtualModel(criterion=loss, optimizer=optimizer)
        model.fit()
        plt.figure(figsize=(10, 8))
        plot_points(model.points, optimizer.lr)  # 绘制参数迭代路径.
        plot_contour(model.criterion, r=15, n=400)  # 绘制等高线.
        plt.show()
