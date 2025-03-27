import numpy as np


def forward(X):
    """前向传播"""
    Z1 = linear(X, W1, b1)
    H1 = heaviside(Z1, 0)
    Z2 = linear(H1, W2, b2)
    y_hat = heaviside(Z2, 0)

    return y_hat


def linear(X, W, b):
    """线性求和"""
    return np.matmul(X, W) + b


def heaviside(z, theta):
    """激活函数"""
    return np.where(z >= theta, 1, 0)


if __name__ == '__main__':
    # 异或问题输入.
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = [0, 1, 1, 0]

    # 设置权重和偏置项.
    W1 = np.ones(shape=(2, 2))
    b1 = np.array([-0.5, -1.5])
    W2 = np.array([1, -1])
    b2 = -0.5

    y_hat = forward(X)
    print(y_hat)
