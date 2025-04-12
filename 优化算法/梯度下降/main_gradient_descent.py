import torch
from common import VirtualModel
from common import SquareFunction
from gradient_descent import GradientDescent

import matplotlib.pyplot as plt
from common import MSELoss, generate_data, plot_contour_with_sample, plot_points, LinearModel

if __name__ == '__main__':
    initial_param = torch.tensor([10.0, 10.0], requires_grad=True)
    loss = SquareFunction(torch.tensor([1.0, 2.0]))
    model = VirtualModel(
        optimizer=GradientDescent(loss=loss, params=initial_param, lr=0.1)
    )
    model.fit()

    # 线性回归.
    X, y = generate_data()  # X:Tensor(10,1), y:Tensor(10,1)
    loss = MSELoss()
    lr = 0.01
    model = LinearModel(
        in_features=1,
        out_features=1,
        optimizer=GradientDescent(loss, params=initial_param, lr=lr)  # 直接传递模型参数
    )
    model.fit(X, y)
    plt.plot(model.loss_history)  # 学习曲线.
    plot_contour_with_sample(loss, X, y, r=11, n=400)  # 绘制等高线.
    plot_points(model.points, lr, annotate=False)  # 绘制参数迭代路径.
    plt.show()
