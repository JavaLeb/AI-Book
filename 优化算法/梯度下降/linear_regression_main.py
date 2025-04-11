import matplotlib.pyplot as plt
import torch
from common import MSELoss, generate_data, plot_contour_with_sample, plot_points, LinearModel
from gradient_descent import GradientDescent

if __name__ == "__main__":
    X, y = generate_data()  # X:Tensor(10,1), y:Tensor(10,1)
    loss = MSELoss()
    init_params = torch.tensor([10.0, 10.0], requires_grad=True)  # shape: [out=1, in=1]
    lr = 0.01
    model = LinearModel(
        in_features=1,
        out_features=1,
        optimizer=GradientDescent(loss,params=init_params, lr=lr)  # 直接传递模型参数
    )
    model.fit(X, y)
    plt.plot(model.loss_history)  # 学习曲线.
    plot_contour_with_sample(loss, X, y, r=11, n=400)  # 绘制等高线.
    plot_points(model.points, lr, annotate=False)  # 绘制参数迭代路径.
    plt.show()
