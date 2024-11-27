import statsmodels.api as sm
from pandas import DataFrame
from algebra_linear_reg import generate_data, split_data
import numpy as np


def train_model(data: DataFrame, const=True):
    """
    训练模型.
    :param data: 训练数据.
    :param const: 是否添加常数项.
    :return: 训练好的模型.
    """

    # 添加常数项.
    if const:
        x = sm.add_constant(data[['x']])
    else:
        x = data[['x']]

    y = data[['y']]
    # 最小二乘法初始化模型.
    model = sm.OLS(y, x)
    # 拟合模型.
    model = model.fit()

    # 输出拟合后的模型摘要信息.
    print(model.summary())

    return model


if __name__ == '__main__':
    # 数据读取.
    data = generate_data()

    # 划分数据.
    train_data, valid_data = split_data(data)

    train_x = train_data['x']
    train_y = train_data['y']
    m = len(train_x)
    train_x_bar = np.average(train_x)
    e = np.sum((train_y - 1.7775 * train_x - 2.1196) ** 2)
    x_e = np.sum((train_x - train_x_bar) ** 2)
    theta_0_err = e * np.sum(train_x ** 2) / (m * (m - 2) * x_e)


    theta_1_err = e / ((m - 2) * x_e)
    print(np.sqrt(theta_0_err))
    print(np.sqrt(theta_1_err))
    # 训练模型：模型有截距项.
    train_model(train_data, const=True)
    # 训练模型：模型没有截距项.
    train_model(train_data, const=False)
