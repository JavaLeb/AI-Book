import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from matplotlib.colors import ListedColormap

from sklearn.utils.extmath import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, roc_auc_score
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import average_precision_score, auc
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
from sklearn.preprocessing import label_binarize

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 显示中文
plt.rcParams["axes.unicode_minus"] = False  # 显示负号


def generate_data():
    centers = [[-5, 0], [0, 1.5], [5, -1]]
    X, y = make_blobs(n_samples=[100, 800, 100], centers=centers, cluster_std=2, random_state=40)
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
    scatter = ax.scatter(x1, x2, y, c=y, edgecolor="black")
    # 添加标题和坐标轴标签
    ax.set_title("样本三维散点图")
    ax.set_xlabel(r"$x_1$", fontsize=12)
    ax.set_ylabel(r"$x_2$", fontsize=12)
    ax.set_zlabel('类别')
    ax.legend(*scatter.legend_elements(), title="类别")

    # plt.show()


def split_data(data: DataFrame):
    """
    划分训练集和验证集.
    :param data: 原始数据.
    :return: 划分好的数据.
    """

    train_data, valid_data = train_test_split(data, train_size=0.7, shuffle=True, random_state=42)

    return train_data, valid_data


def train_model(train_data: DataFrame):
    X = train_data[['x1', 'x2']]
    y = train_data['y']
    model = LogisticRegression()
    model.fit(X, y)

    print('model.classes_=', model.classes_)
    print('model.coef_=\n', np.round(model.coef_, 2))
    print('model.intercept_=', np.round(model.intercept_, 2))

    # 对新样本预测.
    x_new = [[1, 2]]
    y_proba = model.predict_proba(x_new)
    y_predict = model.predict(x_new)
    print('y_proba=', y_proba)
    print('y_predict=', y_predict)

    t = np.sum(model.coef_ * x_new, axis=1) + model.intercept_
    t = t.reshape((1, t.shape[0]))
    user_y_proba = softmax(t)
    print('user_y_proba=', user_y_proba)

    ovr_model = OneVsRestClassifier(LogisticRegression()).fit(X, y)

    ovo_model = OneVsOneClassifier(LogisticRegression()).fit(X, y)

    return model


def evaluate_model(model, data: DataFrame):
    X = data[['x1', 'x2']]  # 特征
    y_true = data['y']  # 目标标签.

    y_predict = model.predict(X)  # 模型预测.
    y_scores = model.predict_proba(X)  # 预测概率.

    cm = confusion_matrix(y_true, y_predict)  # 混淆矩阵.
    accuracy = accuracy_score(y_true, y_predict)  # 准确率.
    beta, retain_n = 2, 3  # 保留小数位.
    model_metric_df = pd.DataFrame()
    averages = ['macro', 'weighted', 'micro']  # 各种指标计算策略.
    for average in averages:  # 计算精准率、召回率、f1、fbeta.
        model_metric_df.loc['precision', average] = precision_score(y_true, y_predict, average=average)
        model_metric_df.loc['recall', average] = recall_score(y_true, y_predict, average=average)
        model_metric_df.loc['f1', average] = f1_score(y_true, y_predict, average=average)
        model_metric_df.loc['fbeta', average] = fbeta_score(y_true, y_predict, beta=beta, average=average)
    # 打印指标.
    print('confusion_matrix=\n', cm)
    print('accuracy=', round(accuracy, retain_n))
    print(model_metric_df.round(retain_n).to_markdown())

    # 可视化混淆矩阵.
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('混淆矩阵')

    binarized_y_true = label_binarize(y_true, classes=model.classes_)  # 二值化真实类别.
    binarized_y_predict = label_binarize(y_predict, classes=model.classes_)  # 二值化预测类别.
    weighted = y_true.value_counts(normalize=True, sort=False)  # 统计每个类别占比，降序排序.
    print('weighted=', weighted)

    cls_metric_df = pd.DataFrame()  # 每个类的评估指标.
    pr_fig = plt.figure()  # 初始化绘制P-R曲线的图.
    pr_ax = pr_fig.add_subplot(111)
    roc_fig = plt.figure()  # 初始化绘制ROC曲线的图.
    roc_ax = roc_fig.add_subplot(111)
    precision_dic, recall_dic, fpr_dic, tpr_dic = dict(), dict(), dict(), dict()
    # 计算各类别的指标.
    for cls in model.classes_:
        cls_y_true, cls_y_predict, cls_y_score = binarized_y_true[:, cls], binarized_y_predict[:, cls], y_scores[:, cls]

        cls_metric_df.loc[cls, 'precision'] = precision_score(cls_y_true, cls_y_predict)  # 每个类的精准率.
        cls_metric_df.loc[cls, 'recall'] = recall_score(cls_y_true, cls_y_predict)  # 每个类的召回率.
        cls_metric_df.loc[cls, 'f1'] = f1_score(cls_y_true, cls_y_predict)  # 每个类的f1-score.
        cls_metric_df.loc[cls, 'fbeta'] = fbeta_score(cls_y_true, cls_y_predict, beta=beta)  # 每个类的fbeta-score.
        cls_average_precision = average_precision_score(cls_y_true, cls_y_score)  # 每个类的AP.
        cls_metric_df.loc[cls, 'average_precision'] = cls_average_precision

        # 计算每个类的阈值对应精准率和召回率.
        # 注意recall是降序的，也就是从P-R曲线最右边的点开始.
        precisions, recalls, thresholds = precision_recall_curve(
            cls_y_true, cls_y_score, drop_intermediate=False)
        precision_dic[cls], recall_dic[cls] = precisions, recalls
        cls_pr_auc = auc(recalls, precisions)  # PR-AUC.
        cls_metric_df.loc[cls, 'pr-auc'] = cls_pr_auc

        fpr, tpr, thresholds = roc_curve(cls_y_true, cls_y_score, drop_intermediate=False)  # 假阳性率，真阳性率，阈值.
        fpr_dic[cls], tpr_dic[cls] = fpr, tpr
        cls_roc_auc_s = roc_auc_score(cls_y_true, cls_y_score)  # ROC AUC，等价于auc.
        cls_roc_auc = auc(fpr, tpr)  # ROC AUC.
        cls_metric_df.loc[cls, 'roc_auc_s'] = cls_roc_auc_s
        cls_metric_df.loc[cls, 'roc_auc'] = cls_roc_auc
        # 绘制每个类的P-R曲线.
        label = f'类别{cls}: AP={round(cls_average_precision, retain_n)}, PR-AUC={round(cls_pr_auc, retain_n)}'
        pr_ax.plot(recalls, precisions, label=label)
        # 绘制ROC曲线.
        label = f'类别{cls}: ROC-AUC={round(cls_roc_auc, retain_n)}'
        roc_ax.plot(fpr, tpr, label=label)

    # 合并所有类的recall，合并结果默认是降序的，使用[::-1]进行反转.
    all_recall = np.unique(np.concatenate([recall_dic[cls] for cls in model.classes_]))[::-1]
    # 合并所有类的fpr.
    all_fpr = np.unique(np.concatenate([fpr_dic[cls] for cls in model.classes_]))
    all_precision_df, all_tpr_df = pd.DataFrame(), pd.DataFrame()
    for cls in model.classes_:
        # 使用每个类的P-R曲线线性插值. 需要将序列使用[::-1]进行反转变成升序.
        all_precision_df[cls] = np.interp(all_recall, recall_dic[cls][::-1], precision_dic[cls][::-1])
        all_tpr_df[cls] = np.interp(all_fpr, fpr_dic[cls], tpr_dic[cls])  # 使用每个类的ROC曲线线性插值.
    # macro P-R曲线.
    macro_precision = np.mean(all_precision_df, axis=1)
    macro_ap = np.sum((all_recall[:-1] - all_recall[1:]) * macro_precision[:-1])  # macro AP.
    macro_pr_auc = auc(all_recall, macro_precision)  # macro PR AUC.
    label = f'Macro: AP={np.round(macro_ap, retain_n)}, PR-AUC={round(macro_pr_auc, retain_n)}'
    pr_ax.plot(all_recall, macro_precision, label=label)
    # weighted P-R曲线.
    weighted_precision = np.sum(all_precision_df * weighted, axis=1)
    weighted_pr_auc = auc(all_recall, weighted_precision)  # macro PR AUC.
    weighted_ap = np.sum((all_recall[:-1] - all_recall[1:]) * weighted_precision[:-1])  # weighted AP.
    label = f'Weighted: AP={np.round(weighted_ap, retain_n)}, PR-AUC={round(weighted_pr_auc, retain_n)}'
    pr_ax.plot(all_recall, weighted_precision, label=label)

    # macro roc曲线.
    macro_tpr = np.mean(all_tpr_df, axis=1)
    macro_roc_auc = auc(all_fpr, macro_tpr)
    label = f'Macro: ROC-AUC={round(macro_roc_auc, retain_n)}'
    roc_ax.plot(all_fpr, macro_tpr, label=label)
    # weighted roc曲线.
    weighted_tpr = np.sum(all_tpr_df * weighted, axis=1)
    weighted_roc_auc = auc(all_fpr, weighted_tpr)
    label = f'Weighted: ROC-AUC={round(weighted_roc_auc, retain_n)}'
    roc_ax.plot(all_fpr, weighted_tpr, label=label)

    pr_ax.legend()
    pr_ax.set_xlabel('召回率Recall')
    pr_ax.set_ylabel('精准率Precision')
    pr_ax.set_title(f'多元逻辑回归Precision-Recall曲线')

    roc_ax.set_xlabel('假阳性率FPR')
    roc_ax.set_ylabel('真阳性率TPR')
    roc_ax.set_title(f'多元逻辑回归ROC曲线')
    roc_ax.legend()

    plt.show()


def visualize_model(data: DataFrame, model):
    X = data[['x1','x2']].values
    y = data['y'].values
    custom_cmap = ListedColormap(["#fafab0", "#9898ff", "#a0faa0"])

    x0, x1 = np.meshgrid(np.linspace(0, 8, 500).reshape(-1, 1),
                         np.linspace(0, 3.5, 200).reshape(-1, 1))
    X_new = np.c_[x0.ravel(), x1.ravel()]

    y_proba = model.predict_proba(X_new)
    y_predict = model.predict(X_new)

    zz1 = y_proba[:, 1].reshape(x0.shape)
    zz = y_predict.reshape(x0.shape)

    plt.figure(figsize=(10, 4))
    plt.plot(X[y == 2, 0], X[y == 2, 1], "g^", label="Iris virginica")
    plt.plot(X[y == 1, 0], X[y == 1, 1], "bs", label="Iris versicolor")
    plt.plot(X[y == 0, 0], X[y == 0, 1], "yo", label="Iris setosa")

    plt.contourf(x0, x1, zz, cmap=custom_cmap)
    contour = plt.contour(x0, x1, zz1, cmap="hot")
    plt.clabel(contour, inline=1)
    plt.xlabel("Petal length")
    plt.ylabel("Petal width")
    plt.legend(loc="center left")
    plt.axis([0.5, 7, 0, 3.5])
    plt.grid()

    plt.show()

    x1 = data['x1']
    x2 = data['x2']
    y = data['y']
    fig, ax = plt.subplots(figsize=(6, 4))

    scatter = ax.scatter(x1, x2, c=y, edgecolor="black")
    ax.set(title="数据样本散点图", xlabel=r"$x_1$", ylabel=r"$x_2$")
    _ = ax.legend(*scatter.legend_elements(), title="类别")

    # 0 = theta_0 + theta_1 * x1 + theta_2 * x2
    func = lambda x, p: (-theta_0 - theta_1 * x - np.log(p / (1 - p))) / theta_2
    x = np.arange(-4, 4, 0.01)
    for i in [0, 1, 2]:
        theta_0 = model.intercept_[i]
        theta_1 = model.coef_[i][0]
        theta_2 = model.coef_[i][1]

        for p in [0.33]:
            plt.plot(x, func(x, p), label=f'P={p}', linestyle='--')
        # 添加标题和坐标轴标签
        plt.title(rf'决策边界${round(theta_0, 2)} + {round(theta_1, 2)} x_1 + {round(theta_2, 2)} x_2$')
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.legend()
    #
    # s_x1 = np.arange(-5, 5, 0.05)
    # s_x2 = np.arange(-5, 5, 0.05)
    # s_x1, s_x2 = np.meshgrid(s_x1, s_x2)
    # t = theta_0 + theta_1 * s_x1 + theta_2 * s_x2
    # z = 1 / (1 + np.exp(-t))
    # # 绘制三维散点图.
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x1[label_1_y], x2[label_1_y], y[label_1_y], color='g', marker='^', label='正样本')
    # ax.scatter(x1[label_0_y], x2[label_0_y], y[label_0_y], color='r', label='负样本')
    # ax.plot(s_x1, s_x2, z, alpha=0.5, label='逻辑回归曲面')
    # # 添加标题和坐标轴标签
    # ax.set_title("逻辑回归模型")
    # ax.set_xlabel("x1", fontsize=12)
    # ax.set_ylabel("x2", fontsize=12)
    # ax.set_zlabel('P')
    # ax.legend()
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # DecisionBoundaryDisplay.from_estimator(model, X, ax=ax, response_method="predict")
    # ax.scatter(x1[label_1_y], x2[label_1_y], color='g', marker='^', label='正样本')
    # ax.scatter(x1[label_0_y], x2[label_0_y], color='r', label='负样本')
    # ax.set_xlabel("x1", fontsize=12)
    # ax.set_ylabel("x2", fontsize=12)
    # ax.set_title('决策边界')
    # ax.legend()

    plt.show()


if __name__ == '__main__':
    data = generate_data()
    # visualize_data(data)
    train_data, valid_data = split_data(data)
    model = train_model(train_data)
    # evaluate_model(model, train_data)
    visualize_model(data, model)
