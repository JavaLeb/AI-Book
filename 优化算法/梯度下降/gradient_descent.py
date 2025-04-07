import torch
import numpy as np


def gradient_descent(func, init_params, lr=0.1, max_iters=100, tol=1e-6):
    """
    梯度下降法.

    :param func: 目标函数（需返回标量）
    :param init_params: 初始参数（Tensor或可迭代对象）
    :param lr: 学习率
    :param max_iters: 最大迭代次数
    :param tol: 收敛阈值（两次迭代损失变化小于此值则停止）
    :return:
    """
    # 确保输入参数为PyTorch张量，并启用梯度追踪
    if not isinstance(init_params, torch.Tensor):
        params = torch.tensor(init_params, dtype=torch.float32, requires_grad=True)
    else:
        params = init_params.clone().detach().requires_grad_(True)

    loss_history = []  # 记录历史损失.
    points = []
    iteration_n = -1
    for i in range(max_iters):
        points.append(params.clone().detach())  # 记录参数每次迭代的值.
        loss = func(params)  # 计算损失.
        loss_history.append(loss.item())  # 记录损失.

        loss.backward()  # 计算梯度.
        iteration_n = i
        with torch.no_grad():
            params -= lr * params.grad  # 更新参数

            # 检查收敛条件
            if i > 0 and abs(loss_history[-1] - loss_history[-2]) < tol:
                break

        params.grad.zero_()  # 清空梯度.
    params = params.detach()
    print(f"Converged at iteration {iteration_n}, "
          f"Optimized parameters: {np.round(params.numpy(), 3)}, "
          f"Final loss: {loss_history[-1]}")

    return params, loss_history, points
