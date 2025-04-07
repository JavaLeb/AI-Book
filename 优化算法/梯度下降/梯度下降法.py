import numpy as np
import matplotlib.pyplot as plt


def gradient_descent(start, learning_rate, iterations):
    """执行梯度下降算法，返回迭代路径"""
    points = [np.array(start, dtype=float)]  # 初始点
    for _ in range(iterations):
        current = points[-1]
        grad = 2 * current  # 梯度为 (2x, 2y)
        next_point = current - learning_rate * grad
        points.append(next_point.copy())
    return np.array(points)


# 生成等高线数据
x = np.linspace(-15, 15, 400)
y = np.linspace(-15, 15, 400)
X, Y = np.meshgrid(x, y)
Z = X ** 2 + Y ** 2

# 绘制等高线
plt.figure(figsize=(10, 8))
contour = plt.contour(X, Y, Z, levels=np.logspace(0, 3, 10), cmap='viridis')
plt.clabel(contour, inline=True, fontsize=8)

# 执行梯度下降（学习率=1，迭代5次）
initial_point = (10, 10)
learning_rate = 0.1
iterations = 15
points = gradient_descent(initial_point, learning_rate, iterations)
print(points)
# 绘制迭代路径
plt.plot(points[:, 0], points[:, 1], 'ro-', markersize=8, linewidth=1, label='Gradient Descent Path')
plt.scatter(points[0, 0], points[0, 1], color='green', s=100, label='Start (0)')
plt.scatter(points[1:, 0], points[1:, 1], color='red', s=50)

# 标注迭代步骤
for i, (x, y) in enumerate(points):
    plt.annotate(i, (x, y), textcoords="offset points", xytext=(0, 10), ha='center', color='blue')

# 设置图形属性
plt.xlim(-15, 15)
plt.ylim(-15, 15)
plt.xlabel('$\\theta_1$')
plt.ylabel('$\\theta_2$')
plt.title('Gradient Descent on $L(\mathbf{\\theta})=\\theta_1^2+\\theta_2^2$' + f'(Learning Rate = {learning_rate})')
plt.legend()
plt.grid(True)
plt.show()
