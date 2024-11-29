import statsmodels.api as sm
from pandas import DataFrame
from algebra_linear_reg import generate_data, split_data
import numpy as np
import scipy.stats as stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
import matplotlib.pyplot as plt


def train_model(data: DataFrame, const=True):
    """
    训练模型.
    :param data: 训练数据.
    :param const: 是否添加常数项.
    :return: 训练好的模型.
    """

    # 添加常数项.
    if const:
        x = sm.add_constant(data['x'])
    else:
        x = data['x']

    y = data['y']
    # 最小二乘法初始化模型.
    model = sm.OLS(y, x)
    # 拟合模型.
    model = model.fit()

    # 输出拟合后的模型摘要信息.
    print(model.summary())

    return model


def evaluate_model(trained_model, data: DataFrame, const=True):
    """
    评估模型.
    :param trained_model: 已训练好的模型.
    :param data: 数据
    :param const: 模型是否含有常数项.
    :return:
    """
    # 添加常数项.
    if const:
        valid_x = sm.add_constant(data['x'])
    else:
        valid_x = data['x']
    valid_y = data['y']

    y_hat = trained_model.predict(valid_x)

    mse = mean_squared_error(valid_y, y_hat)
    mae = mean_absolute_error(valid_y, y_hat)
    rmse = root_mean_squared_error(valid_y, y_hat)

    if const:
        r2 = r2_score(valid_y, y_hat)
    else:
        ss_err = np.sum((y_hat - valid_y) ** 2)
        ss_tot = np.sum(valid_y ** 2)  # 未中心化.
        r2 = 1 - ss_err / ss_tot  # 决定系数.

    print(f'mse={round(mse, 4)}, mae={round(mae, 4)}, rmse={round(rmse, 4)}, r2={round(r2, 4)}')


def visualize_model(data, const_model, model):
    # 添加常数项.
    x = data['x']
    y = data['y']
    predict_x = sm.add_constant(x)
    # 绘制散点图
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 避免散点图显示中文乱码.
    plt.scatter(x, y)  # 绘制样本点.
    const_y_hat = const_model.predict(predict_x)  # 有常数项模型预测.
    const_label = rf'$y = {round(const_model.params["x"], 4)} x + {round(const_model.params["const"], 4)} + \varepsilon $'
    plt.plot(x, const_y_hat, color='red', label=const_label)  # 绘制有常数项模型.
    y_hat = model.predict(x)  # 无常数项模型预测.
    label = rf'$y = {round(model.params["x"], 4)} x + \varepsilon $'
    plt.plot(x, y_hat, color='green', label=label)  # 绘制常数项模型.
    # 添加标题和坐标轴标签
    plt.title("产品个数-生产成本模型")
    plt.xlabel("产品个数")
    plt.ylabel("生产成本")
    plt.legend()

    # 显示图形
    plt.show()


def visualize_residual(data: DataFrame, const_model, model):
    # 添加常数项.
    x = data['x']
    y = data['y']

    predict_x = sm.add_constant(x)

    # 绘制散点图
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 避免散点图显示中文乱码.
    plt.rcParams['axes.unicode_minus'] = False  # 显示坐标轴负号.

    # 含常数模型.
    y_hat = const_model.predict(predict_x)
    const_residual = y - y_hat  # 残差.
    plt.scatter(y_hat, const_residual, color='red', label='含常数模型样本点')  # 绘制样本点.


    # 无常数模型.
    y_hat = model.predict(x)
    residual = y - y_hat  # 残差.
    plt.scatter(y_hat, residual, color='green', label='无常数模型样本点')  # 绘制样本点.

    #  residual=0.
    plt.axhline(y=0.0, color='blue', linewidth=1)


    # 添加标题和坐标轴标签
    plt.title("残差图")
    plt.xlabel("预测值")
    plt.ylabel("残差")
    plt.legend()


    # 残差QQ图.
    # 残差QQ图.
    plt.figure()
    plt.subplot(1,2,1)
    stats.probplot(const_residual,plot=plt)
    plt.title('含常数项模型残差QQ图')

    plt.subplot(1,2,2)
    stats.probplot(residual,plot=plt)
    plt.title('无常数项模型残差QQ图')

    # 显示图形
    plt.show()


def user_defined_stat_model_const(data: DataFrame):
    """
    用户自定义统计模型：模型有常数项.

    :param data: 数据.
    :return:
    """
    data_x = data['x']
    data_y = data['y']

    m = len(data_x)

    x_bar = np.average(data_x)  # x平均值.
    y_bar = np.average(data_y)  # y平均值.
    train_var_x = np.sum((data_x - x_bar) ** 2) / (m - 1)  # 自变量的方差.
    train_cov_x_y = np.sum((data_x - x_bar) * (data_y - y_bar)) / (m - 1)  # 自变量与因变量的协方差.
    # 回归系数.
    theta_1_hat = train_cov_x_y / train_var_x  # theta1的估计值，不受m-1的影响.
    theta_0_hat = np.average(data_y) - theta_1_hat * np.average(data_x)  # theta0的估计值.
    print('theta_0_hat=', round(theta_0_hat, 4))
    print('theta_1_hat=', round(theta_1_hat, 4))
    # 随机误差.
    e_sum = np.sum((data_y - theta_1_hat * data_x - theta_0_hat) ** 2)
    x_bar = np.average(data_x)
    x_e = np.sum((data_x - x_bar) ** 2)

    df = m - 2  # 自由度.
    s_theta_0 = np.sqrt(e_sum * np.sum(data_x ** 2) / (m * df * x_e))
    s_theta_1 = np.sqrt(e_sum / (df * x_e))  # 估计量的标准方差.
    print('s_theta_0=', round(s_theta_0, 3))
    print('s_theta_1=', round(s_theta_1, 3))
    likelihood_sigma_2 = e_sum / m
    sigma_2 = e_sum / df  # 干扰项方差的估计值.
    print('likelihood_sigma_2=', likelihood_sigma_2)
    print('sigma_2=', round(sigma_2, 3))

    # 估计量的t值
    t_theta_0 = theta_0_hat / s_theta_0
    t_theta_1 = theta_1_hat / s_theta_1
    print('t_theta_0=', round(t_theta_0, 3))
    print('t_theta_1=', round(t_theta_1, 3))

    # 示例：df为自由度，t_value为需要计算概率的值.
    # 计算|t|>|t_value|的概率. 它等于 t > |t_value| 和 t < -|t_value| 的概率.
    p0 = 1 - stats.t.cdf(t_theta_0, df) + stats.t.cdf(-t_theta_0, df)
    p1 = 1 - stats.t.cdf(t_theta_1, df) + stats.t.cdf(-t_theta_1, df)
    print(f"P(|t|>t0={round(t_theta_0, 3)}) = {round(p0, 3)}")
    print(f"P(|t|>t1={round(t_theta_1, 3)}) = {round(p1, 3)}")


def user_defined_stat_model(data: DataFrame):
    """
    用户自定义模型：模型没有常数项.
    :param data: 数据.
    :return:
    """

    # 回归系数.
    data_x = data['x']
    data_y = data['y']

    m = len(data_x)
    # 回归系数.
    train_var_x = np.sum(data_x ** 2) / (m - 1)  # 自变量的方差.
    train_cov_x_y = np.sum(data_x * data_y) / (m - 1)  # 自变量与因变量的协方差.
    theta_1_hat = train_cov_x_y / train_var_x  # theta1的估计值.
    print('theta_1_hat=', theta_1_hat)
    # 自由度.
    df = m - 1  # 当theta0=0时，自由度不再是m-2.

    # 随机误差.
    e_sum = np.sum((data_y - theta_1_hat * data_x) ** 2)
    x_e = np.sum(data_x ** 2)
    # 估计量的标准方差.
    s_theta_1 = np.sqrt(e_sum / (df * x_e))
    likelihood_sigma_2 = e_sum / m
    print('likelihood_sigma_2=', likelihood_sigma_2)
    sigma_2 = e_sum / df  # 干扰项方差的估计值.

    print('s_theta_1=', round(s_theta_1, 3))
    print('sigma_2=', round(sigma_2, 3))

    # 估计量的t值
    t_theta_1 = theta_1_hat / s_theta_1
    print('t_theta_1=', round(t_theta_1, 3))

    # df为自由度，t_value为需要计算概率的值
    # 计算|t|>|t_value|的概率. 它等于 t > |t_value| 和 t < -|t_value| 的概率.
    p1 = 1 - stats.t.cdf(t_theta_1, df) + stats.t.cdf(-t_theta_1, df)
    print(f"P(|t|>t1={round(t_theta_1, 3)}) = {round(p1, 3)}")


if __name__ == '__main__':
    # 数据读取.
    data = generate_data()

    # 划分数据.
    train_data, valid_data = split_data(data)

    train_x = train_data['x']
    train_y = train_data['y']

    # 训练模型：模型有截距项.
    const_model = train_model(train_data)
    evaluate_model(const_model, train_data)
    evaluate_model(const_model, valid_data)
    user_defined_stat_model_const(train_data)

    # 训练模型：模型没有截距项.
    model = train_model(train_data, const=False)
    # 模型评估.
    evaluate_model(model, train_data, const=False)
    evaluate_model(model, valid_data, const=False)
    user_defined_stat_model(train_data)

    # 可视化模型.
    # visualize_model(data, const_model, model)

    visualize_residual(data, const_model, model)
