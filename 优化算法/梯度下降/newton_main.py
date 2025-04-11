import torch
from common import VirtualModel, SquareFunction, plot_points, plot_contour
from gradient_descent import NewtonGradient
import matplotlib.pyplot as plt

if __name__ == "__main__":
    initial_params = torch.tensor([10.0, 10.0], requires_grad=True)
    plt.figure(figsize=(10, 8))
    loss = SquareFunction(torch.tensor([1.0, 2.0]))
    lr = 1
    model = VirtualModel(
        optimizer=NewtonGradient(loss, params=initial_params, lr=lr)
    )
    model.fit()
    plot_points(model.points, lr)  # 绘制参数迭代路径.
    plot_contour(loss, r=15, n=400)  # 绘制等高线.
    plt.show()
