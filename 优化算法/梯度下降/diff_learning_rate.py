import torch
from virtualmodel import VirtualModel
from loss import Loss
from gradient_descent import GradientDescent
from plot import plot_points, plot_contour
import matplotlib.pyplot as plt

if __name__ == "__main__":

    initial_params = [torch.tensor([10.0, 10.0], requires_grad=True),
                      torch.tensor([-10.0, 10.0], requires_grad=True),
                      torch.tensor([-10.0, -10.0], requires_grad=True)]
    learning_rates = [0.5, 0.45, 0.1]
    plt.figure(figsize=(10, 8))
    loss = Loss()
    for i in range(len(learning_rates)):
        lr = learning_rates[i]
        model = VirtualModel(
            loss=loss,
            optimizer=GradientDescent(params=[initial_params[i]], lr=lr)
        )
        model.fit()
        plot_points(model.points, lr)  # 绘制参数迭代路径.
    plot_contour(loss,r=15,n=400)  # 绘制等高线.
    plt.show()
