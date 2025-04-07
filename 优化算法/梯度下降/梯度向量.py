import numpy as np
import matplotlib.pyplot as plt


def f(x, y):
    return x ** 2 + 2 * y ** 2


# 梯度计算函数
def gradient(x, y):
    return 2 * x, 4 * y


# 生成曲面数据
r = 3
x = np.linspace(-r, r, 100)
y = np.linspace(-r, r, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# 定义点和梯度
points = [(1, 1), (-1, -0.5)]
gradients = [gradient(p[0], p[1]) for p in points]


# 切平面方程和局部网格
def generate_tangent_plane(px, py, range=0.5):
    # 生成局部网格（缩小范围）
    x_t = np.linspace(px - range, px + range, 10)
    y_t = np.linspace(py - range, py + range, 10)
    X_t, Y_t = np.meshgrid(x_t, y_t)
    Z_t = px ** 2 + py ** 2 + 2 * px * (X_t - px) + 2 * py * (Y_t - py)
    return X_t, Y_t, Z_t


# 绘制图形
fig = plt.figure(figsize=(20, 10))

# --------------------------
# 1. 3D曲面、梯度、切平面与方向向量
# --------------------------
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)

# 为每个点添加梯度和切平面
for (px, py) in points:
    gx, gy = gradient(px, py)
    pz = f(px, py)

    # 局部切平面（橙色，范围缩小）
    X_t, Y_t, Z_t = generate_tangent_plane(px, py, range=2)
    ax1.plot_surface(X_t, Y_t, Z_t, alpha=0.4, color='orange')

    # 计算方向向量（三维切向量）
    grad = gradient(px, py)
    dz = np.dot(grad, grad)  # 方向导数最大值
    vec_scale = 0.5  # 向量缩放因子
    direction = np.array([gx, gy, dz]) * vec_scale

    # 标记点
    ax1.scatter(px, py, pz, color='red', s=50)
    # 绘制方向向量
    ax1.quiver(px, py, pz,
               direction[0], direction[1], direction[2],
               color='red', linewidth=2, arrow_length_ratio=0.1)

ax1.set_xlabel('$x_1$')
ax1.set_ylabel('$x_2$')
ax1.set_zlabel('f')
ax1.set_title('Surface with Gradients, Tangent Planes, and Direction Vectors')

# --------------------------
# 2. 等高线、梯度投影与方向向量
# --------------------------
ax2 = fig.add_subplot(122)

# 等高线
contour = ax2.contour(X, Y, Z, levels=np.arange(0, 10, 1), cmap='viridis')
ax2.clabel(contour, inline=True, fontsize=8)

# 梯度投影与方向向量
scale = 0.5  # 箭头缩放因子
for (px, py), (gx, gy) in zip(points, gradients):
    # 梯度投影（红色）
    ax2.quiver(px, py, gx * scale, gy * scale, color='r', angles='xy', scale_units='xy', scale=1)
    ax2.scatter(px, py, color='red', s=50)

    # 方向向量示例（蓝色）
    ax2.quiver(px, py, 1, 0, color='b', angles='xy', scale_units='xy', scale=1, alpha=0.5)  # 沿x轴
    ax2.quiver(px, py, 0, 1, color='b', angles='xy', scale_units='xy', scale=1, alpha=0.5)  # 沿y轴

ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')
ax2.set_title('Contour Projections Plot with Gradient and Direction Vectors')
ax2.grid(True)
ax2.axis('equal')

plt.tight_layout()
plt.show()
# 等高线图及梯度投影
ax2 = fig.add_subplot(122)
contour = ax2.contour(X, Y, Z, levels=np.arange(0, 20, 2), cmap='viridis')
ax2.clabel(contour, inline=True, fontsize=8)
ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')
ax2.set_title('Contour Projections Plot with Gradient')
ax2.grid(True)

# 在等高线图上添加梯度向量（投影到XY平面）
scale = 0.3  # 调整箭头长度
for (px, py), (gx, gy) in zip(points, gradients):
    ax2.quiver(px, py, gx * scale, gy * scale, color='r', angles='xy', scale_units='xy', scale=1)
    ax2.scatter(px, py, color='red', s=50)  # 标记点

plt.tight_layout()
plt.show()
