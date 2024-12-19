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
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay, classification_report
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

    plt.show()


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

    # 训练OvR模型.
    ovr_model = OneVsRestClassifier(LogisticRegression()).fit(X, y)

    return model, ovr_model


def evaluate_model(model, data: DataFrame, model_name, data_name):
    title = f'{data_name}-{model_name}'

    X = data[['x1', 'x2']]  # 特征
    y_true = data['y']  # 目标标签.

    y_predict = model.predict(X)  # 模型预测.
    y_scores = model.predict_proba(X)  # 预测概率.

    cm = confusion_matrix(y_true, y_predict)  # 混淆矩阵.
    cr = classification_report(y_true, y_predict)
    accuracy = accuracy_score(y_true, y_predict)  # 准确率.
    beta, retain_n = 2, 3  # 保留小数位.
    model_metric_df = pd.DataFrame()
    averages = ['macro', 'weighted', 'micro']  # 各种指标计算策略.
    for average in averages:  # 计算精准率、召回率、f1、fbeta.
        model_metric_df.loc['precision', average] = precision_score(y_true, y_predict, average=average)
        model_metric_df.loc['recall', average] = recall_score(y_true, y_predict, average=average)
        model_metric_df.loc['accuracy', average] = model_metric_df.loc['recall', average]
        model_metric_df.loc['f1', average] = f1_score(y_true, y_predict, average=average)
        model_metric_df.loc['fbeta', average] = fbeta_score(y_true, y_predict, beta=beta, average=average)

    # 可视化混淆矩阵.
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f'{title} 混淆矩阵')

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
        cls_metric_df.loc['accuracy', cls] = accuracy_score(cls_y_true[cls_y_true==1], cls_y_predict[cls_y_true==1])  # 每个类的准确率.
        cls_metric_df.loc['precision', cls] = precision_score(cls_y_true, cls_y_predict)  # 每个类的精准率.
        cls_metric_df.loc['recall', cls] = recall_score(cls_y_true, cls_y_predict)  # 每个类的召回率.
        cls_metric_df.loc['f1', cls] = f1_score(cls_y_true, cls_y_predict)  # 每个类的f1-score.
        cls_metric_df.loc['fbeta'] = fbeta_score(cls_y_true, cls_y_predict, beta=beta)  # 每个类的fbeta-score.
        cls_average_precision = average_precision_score(cls_y_true, cls_y_score)  # 每个类的AP.
        cls_metric_df.loc['average_precision', cls] = cls_average_precision

        # 计算每个类的阈值对应精准率和召回率.
        # 注意recall是降序的，也就是从P-R曲线最右边的点开始.
        precisions, recalls, thresholds = precision_recall_curve(
            cls_y_true, cls_y_score, drop_intermediate=False)
        precision_dic[cls], recall_dic[cls] = precisions, recalls
        cls_pr_auc = auc(recalls, precisions)  # PR-AUC.
        cls_metric_df.loc['pr-auc', cls] = cls_pr_auc

        fpr, tpr, thresholds = roc_curve(cls_y_true, cls_y_score, drop_intermediate=False)  # 假阳性率，真阳性率，阈值.
        fpr_dic[cls], tpr_dic[cls] = fpr, tpr
        cls_roc_auc_s = roc_auc_score(cls_y_true, cls_y_score)  # ROC AUC，等价于auc.
        cls_roc_auc = auc(fpr, tpr)  # ROC AUC.
        cls_metric_df.loc['roc_auc_s', cls] = cls_roc_auc_s
        cls_metric_df.loc['roc_auc', cls] = cls_roc_auc
        # 绘制每个类的P-R曲线.
        label = f'类别{cls}: AP={round(cls_average_precision, retain_n)}, PR-AUC={round(cls_pr_auc, retain_n)}'
        pr_ax.plot(recalls, precisions, label=label)
        # 绘制ROC曲线.
        label = f'类别{cls}: ROC-AUC={round(cls_roc_auc, retain_n)}'
        roc_ax.plot(fpr, tpr, label=label)

    #

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

    model_metric_df.loc['average-precision', 'macro'] = macro_ap
    model_metric_df.loc['average-precision', 'weighted'] = weighted_ap
    model_metric_df.loc['pr-auc', 'macro'] = macro_pr_auc
    model_metric_df.loc['pr-auc', 'weighted'] = weighted_pr_auc
    model_metric_df.loc['roc-auc', 'macro'] = macro_roc_auc
    model_metric_df.loc['roc-auc', 'weighted'] = weighted_roc_auc

    # 打印指标.
    print('confusion_matrix=\n', cm)
    print('class_report=\n', cr)
    print('accuracy=', round(accuracy, retain_n))
    print('model_metric_df=\n', model_metric_df.round(retain_n).to_markdown())
    print('cls_metric_df=\n', cls_metric_df.round(retain_n).to_markdown())

    pr_ax.legend()
    pr_ax.set_xlabel('召回率Recall')
    pr_ax.set_ylabel('精准率Precision')
    pr_ax.set_title(f'{title} Precision-Recall曲线')

    roc_ax.set_xlabel('假阳性率FPR')
    roc_ax.set_ylabel('真阳性率TPR')
    roc_ax.set_title(f'{title} ROC曲线')
    roc_ax.legend()

    plt.show()


def visualize_model(data: DataFrame, model):
    x1 = data['x1']
    x2 = data['x2']
    y = data['y']

    # 制作网格，将网格上的点看做新的样本.
    mesh_x1, mesh_x2 = np.meshgrid(np.linspace(-6, 6, 500).reshape(-1, 1),
                                   np.linspace(-10, 10, 200).reshape(-1, 1))
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
    custom_cmap = ListedColormap(["#fafab0", "#9898ff", "#a0faa0"])
    ax.contourf(mesh_x1, mesh_x2, y_predict, cmap=custom_cmap)
    # 使用网格点的预测概率绘制等概率线，同一等概率线上的概率相等
    contour = ax.contour(mesh_x1, mesh_x2, proba_z, cmap="hot")
    # 设置等概率线.
    ax.clabel(contour, inline=0.1, fontsize=12)
    # 绘制数据样本.
    scatter = ax.scatter(x1, x2, c=y, edgecolor="black")
    ax.legend(*scatter.legend_elements(), title="类别")
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_title('决策边界')
    plt.show()


if __name__ == '__main__':
    # 生成数据.
    data = generate_data()
    # 可视化数据.
    visualize_data(data)
    # 切分数据.
    train_data, valid_data = split_data(data)
    # 训练模型.
    model, ovr_model = train_model(train_data)
    # 模型评估.
    evaluate_model(model, train_data, 'softmax', '训练集')  # 在训练集上评估softmax回归.
    evaluate_model(model, valid_data, 'softmax', '验证集')  # 在验证集上评估softmax回归.
    evaluate_model(ovr_model, train_data, 'OvR', '训练集')  # 在训练集上评估OvR回归.
    evaluate_model(ovr_model, valid_data, 'OvR', '验证集')  # 在验证集上评估OvR回归.
    visualize_model(data, model)  # 可视化softmax回归.
    visualize_model(data, ovr_model)  # 可视化OvR.
