import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score


def generate_data():
    """
    使用 y = 1.9 x + e 产生模拟数据. e是标准正态分布.
    :return: DataFrame 产生的数据.
    """

    # 设置随机数的种子，保证每次运行程序，生成的随机数相同.
    np.random.seed(42)
    # 生成样本点
    start = 10  # 第一个样本点的手机数量.
    m = 20  # 样本点个数.
    x = np.arange(start=start, stop=start + m)
    y = 1.9 * x + np.random.randn(m)  # 加入一些噪声
    y = np.round(y, 2)  # 保留2位小数.
    data = pd.DataFrame()
    data['x'] = x
    data['y'] = y

    return data


def visualize_data(data: DataFrame):
    """
    可视化样本数据.
    :param data: 可视化的样本数据.
    :return:
    """

    # 绘制散点图
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 避免散点图显示中文乱码.
    plt.scatter(data['x'], data['y'])
    # 添加标题和坐标轴标签
    plt.title("产品个数-生产成本散点图")
    plt.xlabel("产品个数")
    plt.ylabel("生产成本")

    # 显示图形
    plt.show()


def split_data(data: DataFrame):
    """
    划分训练集和验证集.
    :param data: 原始数据.
    :return: 划分好的数据.
    """

    train_data = data[:16]  # 前16个样本点作为训练集.
    valid_data = data[16:]  # 剩余的样本点作为验证集.

    return train_data, valid_data


def train_model(data: DataFrame):
    """
    训练模型
    :param data: 训练数据集.
    :return: 训练好的模型
    """
    model = LinearRegression()
    model.fit(data[['x']], data[['y']])
    print(f'coef={model.coef_}, intercept={model.intercept_}')
    theta_1 = model.coef_[0][0]
    theta_0 = model.intercept_[0]
    print(f'y_hat = {round(theta_1, 4)} x + {round(theta_0, 4)}')

    return model


def evaluate_model(trained_model, data: DataFrame):
    valid_x = data[['x']]
    valid_y = data[['y']]
    y_hat = trained_model.predict(valid_x)
    mse = mean_squared_error(valid_y, y_hat)
    mae = mean_absolute_error(valid_y, y_hat)
    rmse = root_mean_squared_error(valid_y, y_hat)
    r2 = r2_score(valid_y, y_hat)
    print(f'mse={round(mse, 4)}, mae={round(mae, 4)}, rmse={round(rmse, 4)}, r2={round(r2, 4)}')


def visualize_model(data, model):
    # 绘制散点图
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 避免散点图显示中文乱码.
    x = data[['x']]
    y = data[['y']]
    plt.scatter(x, y)
    plt.plot(x, model.predict(x), color='red')
    # 添加标题和坐标轴标签
    plt.title("产品个数-生产成本散点图")
    plt.xlabel("产品个数")
    plt.ylabel("生产成本")

    # 显示图形
    plt.show()


def user_defined_linear_reg():
    # 生成数据.
    data = generate_data()

    # 划分训练集和验证集.
    train_data, valid_data = split_data(data)
    train_x = train_data['x']
    train_y = train_data['y']
    valid_x = valid_data['x']
    valid_y = valid_data['y']

    train_x_bar = np.average(train_x)  # 自变量平均值.
    train_y_bar = np.average(train_y)  # 因变量平均值.
    m = len(train_x)
    train_var_x = np.sum((train_x - train_x_bar) ** 2) / (m - 1)  # 自变量的方差.
    train_cov_x_y = np.sum((train_x - train_x_bar) * (train_y - train_y_bar)) / (m - 1)  # 自变量与因变量的协方差.
    theta_1 = train_cov_x_y / train_var_x  # 计算未知参数theta_1.
    theta_0 = train_y_bar - theta_1 * train_x_bar  # 计算未知参数theta_0.
    print(f'x_bar={round(train_x_bar, 2)}, y_bar={round(train_y_bar, 2)}, var_x={round(train_var_x, 2)}, '
          f'cov_x_y={round(train_cov_x_y, 2)}, theta_1={round(theta_1, 4)}, theta_0={round(theta_0, 4)}')

    # 线性模型表达式.
    y_hat_1 = theta_1 * train_x[0] + theta_0
    print('y_hat_1=', round(y_hat_1, 2))  # 预测第一个样本的生成成本.
    # 预测新样本的生产成本.
    x_new = 30
    y_hat_new = theta_1 * x_new + theta_0
    print('y_hat_new=', round(y_hat_new, 2))

    # 自定义实现评估指标.
    valid_y_hat = theta_1 * valid_x + theta_0
    mse = np.average((valid_y_hat - valid_y) ** 2)  # 均方误差.
    mae = np.average(abs(valid_y_hat - valid_y))  # 平均绝对值误差
    rmse = np.sqrt(mse)  # 均方根误差
    valid_y_bar = np.average(valid_y)
    ss_err = np.sum((valid_y_hat - valid_y) ** 2)
    ss_tot = np.sum((valid_y - valid_y_bar) ** 2)
    # 这里使用ss_err，因为模型是使用训练集训练的，ss_tot = ss_reg + ss_err在训练集上成立，
    # 而在验证集上 ss_tot = ss_reg + ss_err 并不成立.
    r2 = 1 - ss_err / ss_tot  # 决定系数.
    print(f'mse={round(mse, 4)}, mae={round(mae, 4)}, rmse={round(rmse, 4)}, r2={round(r2, 4)}')


if __name__ == '__main__':
    user_defined_linear_reg()

    # 生成模拟数据.
    data = generate_data()
    print('data=\n', data)

    # 可视化数据.
    visualize_data(data)

    # 划分训练集和验证集.
    train_data, valid_data = split_data(data)

    # 训练模型.
    model = train_model(train_data)

    # 验证模型.
    evaluate_model(model, valid_data)

    # 可视化模型.
    visualize_model(data, model)
