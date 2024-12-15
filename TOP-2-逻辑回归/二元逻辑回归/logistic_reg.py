import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, fbeta_score, \
    precision_recall_curve, roc_auc_score, roc_curve, ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay, \
    auc


def generate_data():
    """
    :return: DataFrame 产生的数据.
    """

    np.random.seed(40)  # 设置随机种子以获得可复现的结果
    c = [[0], [1]]  # 类别.
    features = 2  # 特征数.
    X = np.r_[5 * np.random.randn(10, features) - 3, 5 * np.random.randn(10, features) + 3]
    X = np.round(X, 2)
    # 每个样本生成对应的类别.
    y = np.array(c[0] * 10 + c[1] * 10)

    data = pd.DataFrame()
    data['x1'] = X[:, 0]
    data['x2'] = X[:, 1]
    data['y'] = y

    shuffled_data = data.sample(frac=1).reset_index(drop=True)

    return shuffled_data


def visualize_data(data: DataFrame):
    """
    可视化样本数据.
    :param data: 可视化的样本数据.
    :return:
    """
    x1 = data['x1']
    x2 = data['x2']
    y = data['y']
    neg = (y == 0)
    pos = (y == 1)

    # 绘制散点图
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 避免散点图显示中文乱码.
    plt.rcParams['axes.unicode_minus'] = False  # 显示坐标轴负号.

    # 绘制散点图
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x1[pos], x2[pos], color='g', marker='^', label='是否购房：1')
    ax.scatter(x1[neg], x2[neg], color='r', label='是否购房：0')
    # 添加标题和坐标轴标签
    ax.set_title("样本二维散点图")
    ax.set_xlabel("新楼盘增长率", fontsize=12)
    ax.set_ylabel("年收入增长率", fontsize=12)
    ax.legend()

    # 绘制三维散点图.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1[pos], x2[pos], y[pos], color='g', marker='^', label='是否购房：1')
    ax.scatter(x1[neg], x2[neg], y[neg], color='r', label='是否购房：0')
    # 添加标题和坐标轴标签
    ax.set_title("样本三维散点图")
    ax.set_xlabel("新楼盘增长率", fontsize=12)
    ax.set_ylabel("年收入增长率", fontsize=12)
    ax.set_zlabel('是否购房')
    ax.legend()

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
    X = data[['x1', 'x2']]
    y = data['y']

    model = LogisticRegression()  # 逻辑回归模型.
    model.fit(X, y)

    print('model.coef_=', model.coef_)  # 模型系数.
    print('model.intercept_=', model.intercept_)  # 模型截距

    return model


def evaluate_model(data: DataFrame, model):
    X = data[['x1', 'x2']]  # 特征
    y = data['y']  # 目标标签.
    y_predict = model.predict(X)  # 模型预测.
    y_scores = model.predict_proba(X)[:, 1]  # 预测概率.
    cm = confusion_matrix(y, y_predict)  # 混淆矩阵.
    accuracy = accuracy_score(y, y_predict)  # 准确率.
    precision = precision_score(y, y_predict)  # 精准率.
    recall = recall_score(y, y_predict)  # 召回率.
    f1 = f1_score(y, y_predict)  # F1 score。
    fbeta = fbeta_score(y, y_predict, beta=2)  # F beta score.
    fpr, tpr, thresholds = roc_curve(y, y_scores, drop_intermediate=False)  # 假阳性率，真阳性率，阈值.
    roc_auc = auc(fpr, tpr)  # AUC.

    print('confusion_matrix=\n', cm)
    print('accuracy=', round(accuracy, 2))
    print('precision=', round(precision, 2))
    print('recall=', round(recall, 2))
    print('f1=', round(f1, 2))
    print(f'fbeta(beta={2})=', round(fbeta, 2))
    print('auc=', round(roc_auc, 2))

    # 可视化混淆矩阵.
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('混淆矩阵')
    # 绘制pr曲线.
    precision, recall, thresholds = precision_recall_curve(y, y_scores)
    pr_display = PrecisionRecallDisplay(precision=precision, recall=recall)
    pr_display.plot()
    plt.scatter(recall, precision, color='red')
    plt.title('Precision-Recall曲线')
    # 绘制 roc曲线
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    display.plot()
    plt.scatter(fpr, tpr, color='red')
    plt.title('ROC曲线')

    plt.show()


def visualize_model(data: DataFrame, model):
    X= data[['x1','x2']].values
    x1 = data['x1']
    x2 = data['x2']
    y = data['y']
    neg = (y == 0)
    pos = (y == 1)

    # 绘制散点图
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 避免散点图显示中文乱码.
    plt.rcParams['axes.unicode_minus'] = False  # 显示坐标轴负号.

    # 绘制散点图

    plt.scatter(x1[pos], x2[pos], color='g', marker='^', label='正样本')
    plt.scatter(x1[neg], x2[neg], color='r', label='负样本')

    x = np.arange(-15, 15, 0.01)
    theta_0 = model.intercept_[0]
    theta_1 = model.coef_[0][0]
    theta_2 = model.coef_[0][1]

    # 0 = theta_0 + theta_1 * x1 + theta_2 * x2
    p = 0.5
    t = (-theta_0 - theta_1 * x + np.log(p / (1 - p))) / theta_2
    plt.plot(x, t, label=f'P={p}',linestyle='--')
    p = 0.9
    t = (-theta_0 - theta_1 * x + np.log(p / (1 - p))) / theta_2
    plt.plot(x, t, label=f'P={p}',linestyle='--')
    p = 0.1
    t = (-theta_0 - theta_1 * x + np.log(p / (1 - p))) / theta_2
    plt.plot(x, t, label=f'P={p}',linestyle='--')
    # 添加标题和坐标轴标签
    plt.title(rf'决策边界${round(theta_0, 2)} + {round(theta_1, 2)} x_1 + {round(theta_2, 2)} x_2$')
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()

    s_x1 = np.arange(-15, 15, 0.05)
    s_x2 = np.arange(-15, 15, 0.05)
    s_x1, s_x2 = np.meshgrid(s_x1, s_x2)
    t = theta_0 + theta_1 * s_x1 + theta_2 * s_x2
    z = 1 / (1 + np.exp(-t))
    # 绘制三维散点图.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1[pos], x2[pos], y[pos], color='g', marker='^', label='正样本')
    ax.scatter(x1[neg], x2[neg], y[neg], color='r', label='负样本')
    ax.plot(s_x1, s_x2, z, alpha=0.5, label='逻辑回归曲面')
    # 添加标题和坐标轴标签
    ax.set_title("逻辑回归模型")
    ax.set_xlabel("x1", fontsize=12)
    ax.set_ylabel("x2", fontsize=12)
    ax.set_zlabel('P')
    ax.legend()

    from sklearn.inspection import DecisionBoundaryDisplay
    plt.figure()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)


    DecisionBoundaryDisplay.from_estimator(
        model,
        X,
        ax=ax1,
        response_method="predict",
        alpha=0.8,
    )
    # scatter = ax1.scatter(X[:, 0], X[:, 1], c= y, edgecolor="k")
    ax1.scatter(x1[pos], x2[pos], color='g', marker='^', label='正样本')
    ax1.scatter(x1[neg], x2[neg], color='r', label='负样本')
    # legend = ax1.legend(*scatter.legend_elements(), title="Classes")
    # ax1.add_artist(legend)
    ax1.set_title('决策边界')

    plt.show()


if __name__ == '__main__':
    data = generate_data()

    visualize_data(data)

    train_data, valid_data = split_data(data)

    model = train_model(train_data)

    # evaluate_model(train_data, model)
    # evaluate_model(valid_data, model)

    visualize_model(data, model)
