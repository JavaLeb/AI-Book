import torch

x1 = torch.tensor(1.0)  # 定义输入和权重
x2 = torch.tensor(1.0)
w1 = torch.tensor(0.0, requires_grad=True)  # 需要梯度追踪
w2 = torch.tensor(1.0, requires_grad=True)

h = (w1 * x1 + w2 * x2) ** 2  # 前向计算
print('输出值=', h)

h.backward()  # 反向传播计算梯度

# 输出梯度
print(f"梯度=({w1.grad}, {w2.grad})")
