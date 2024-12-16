import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
import pandas as pd
from sklearn.datasets import make_blobs

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 显示中文
plt.rcParams["axes.unicode_minus"] = False  # 显示负号


def generate_data():
    centers = [[-5, 0], [0, 1.5], [5, -1]]
    X, y = make_blobs(n_samples=1_000, centers=centers, random_state=40)
    transformation = [[0.4, 0.2], [-0.4, 1.2]]
    X = np.dot(X, transformation)

    data = pd.DataFrame()
    data['x1'] = X[:, 0]
    data['x2'] = X[:, 1]
    data['y'] = y

    print('每个类的样本点个数：\n', data['y'].value_counts())

    return data


def visualize_data(data: DataFrame):
    x1 = data['x1']
    x2 = data['x2']
    y = data['y']
    fig, ax = plt.subplots(figsize=(6, 4))

    scatter = ax.scatter(x1, x2, c=y, edgecolor="black")
    ax.set(title="数据样本散点图", xlabel=r"$x_1$", ylabel=r"$x_2$")
    _ = ax.legend(*scatter.legend_elements(), title="类别")

    # 绘制三维散点图.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter=ax.scatter(x1, x2, y, c=y, edgecolor="black")
    # 添加标题和坐标轴标签
    ax.set_title("样本三维散点图")
    ax.set_xlabel(r"$x_1$", fontsize=12)
    ax.set_ylabel(r"$x_2$", fontsize=12)
    ax.set_zlabel('类别')
    ax.legend(*scatter.legend_elements(), title="类别")

    plt.show()


if __name__ == '__main__':
    data = generate_data()
    visualize_data(data)
