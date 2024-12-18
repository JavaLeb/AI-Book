import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from matplotlib.colors import ListedColormap

from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, roc_auc_score
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import average_precision_score, auc
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

# 绘制散点图
plt.rcParams['font.sans-serif'] = ['SimHei']  # 避免散点图显示中文乱码.
plt.rcParams['axes.unicode_minus'] = False  # 显示坐标轴负号.


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

    print('生成的数据=\n', shuffled_data)

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
    # plt.show()


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
    y_true = data['y']  # 目标标签.

    y_predict = model.predict(X)  # 模型预测.
    y_scores = model.predict_proba(X)  # 预测概率.
    pos_score = y_scores[:, 1]  # 正样本的概率.

    # PR曲线手工绘制测试数据.
    pr_df = pd.DataFrame()
    pr_df['label'] = y_true.copy().values
    pr_df['proba'] = pos_score
    pr_df = pr_df.sort_values(by='proba', ascending=False)
    print(pr_df.round(3))
    model_metric_df = pd.DataFrame()
    cm = confusion_matrix(y_true, y_predict)  # 混淆矩阵.
    model_metric_df['accuracy'] = [accuracy_score(y_true, y_predict)]  # 准确率.
    model_metric_df['precision'] = precision_score(y_true, y_predict)  # 精准率.
    model_metric_df['recall'] = recall_score(y_true, y_predict)  # 召回率.
    model_metric_df['average_precision'] = average_precision_score(y_true, pos_score)  # Average Precision.

    # 计算每个阈值对应的精准率和召回率.
    precisions, recalls, thresholds = precision_recall_curve(y_true, pos_score)
    model_metric_df['pr_auc'] = auc(recalls, precisions)  # P-R曲线下的面积.
    # 自定义计算Average Precision, 与average_precision_score等价.
    user_average_precision = np.sum((recalls[:-1] - recalls[1:]) * precisions[:-1])

    model_metric_df['f1'] = f1_score(y_true, y_predict)  # F1 score.
    model_metric_df['fbeta'] = fbeta_score(y_true, y_predict, beta=2)  # F beta score.
    fpr, tpr, thresholds = roc_curve(y_true, pos_score, drop_intermediate=False)  # 假阳性率，真阳性率，阈值.
    roc_auc = auc(fpr, tpr)  # ROC曲线下的面积, 与roc_auc_score等价.
    model_metric_df['roc_auc'] = roc_auc
    model_metric_df['roc_auc_s'] = roc_auc_score(y_true, pos_score)  # ROC曲线下的面积.

    retain_n = 3  # 保留小数位的个数.
    print('confusion_matrix=\n', cm)
    print('model_metirc=\n', model_metric_df.round(retain_n).to_markdown())

    # 可视化混淆矩阵.
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('混淆矩阵')

    # 绘制pr曲线.
    pr_display = PrecisionRecallDisplay(precision=precisions, recall=recalls)
    pr_display.plot()
    plt.scatter(recalls, precisions, color='red')
    plt.xlabel('召回率Recall', fontsize=12)
    plt.ylabel('精准率Precision', fontsize=12)
    plt.title('Precision-Recall曲线')

    # 绘制 roc曲线
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    display.plot()
    plt.scatter(fpr, tpr, color='red')
    plt.xlabel('假阳性率False Positive Rate', fontsize=12)
    plt.ylabel('真阳性率True Positive Rate', fontsize=12)
    plt.title('ROC曲线')

    plt.show()


def visualize_model(data: DataFrame, model):
    # 绘制散点图
    x1 = data['x1']
    x2 = data['x2']
    y = data['y']
    neg = (y == 0)
    pos = (y == 1)

    # 制作网格，将网格上的点看做新的样本.
    mesh_x1, mesh_x2 = np.meshgrid(np.linspace(-15, 15, 500).reshape(-1, 1),
                                   np.linspace(-12, 12, 200).reshape(-1, 1))
    X_new = np.c_[mesh_x1.ravel(), mesh_x2.ravel()]
    # 预测网格点的概率和类别.
    y_proba = model.predict_proba(X_new)
    y_predict = model.predict(X_new)

    # 网格点属于类别1的概率和归属类.
    proba_z = y_proba[:, 1].reshape(mesh_x1.shape)
    y_predict = y_predict.reshape(mesh_x1.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 使用网格点的类别绘制类别区域，颜色相同的区域类别相同.
    custom_cmap = ListedColormap(["#fafab0", "#9898ff"])
    ax.contourf(mesh_x1, mesh_x2, y_predict, cmap=custom_cmap)
    # 使用网格点的预测概率绘制等概率线，同一等概率线上的概率相等
    contour = ax.contour(mesh_x1, mesh_x2, proba_z, cmap="hot")
    # 设置等概率线.
    ax.clabel(contour, inline=0.1, fontsize=12)
    # 绘制数据样本.
    ax.scatter(x1[pos], x2[pos], color='g', marker='^', label='正样本')
    ax.scatter(x1[neg], x2[neg], color='r', label='负样本')
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_title('决策边界')
    ax.legend()
    plt.show()

    # 绘制三维散点图.
    theta_0 = model.intercept_[0]
    theta_1 = model.coef_[0][0]
    theta_2 = model.coef_[0][1]

    s_x1 = np.arange(-15, 15, 0.05)
    s_x2 = np.arange(-15, 15, 0.05)
    s_x1, s_x2 = np.meshgrid(s_x1, s_x2)
    t = theta_0 + theta_1 * s_x1 + theta_2 * s_x2
    z = 1 / (1 + np.exp(-t))

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

    plt.show()


if __name__ == '__main__':
    data = generate_data()

    # visualize_data(data)

    train_data, valid_data = split_data(data)

    model = train_model(train_data)

    # evaluate_model(train_data, model)
    # evaluate_model(valid_data, model)

    visualize_model(data, model)
