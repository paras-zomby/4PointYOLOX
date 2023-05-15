import itertools
# 相关模块的导入
import sys

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as m
import torch

# numpy数据输出设置
np.set_printoptions(suppress=True, precision=4)


def _set_lim(xlim, ylim):
    if isinstance(xlim, int) or isinstance(xlim, float):
        plt.xlim([0, xlim])
    elif xlim is not None:
        plt.xlim(list(xlim))
    if isinstance(ylim, int) or isinstance(ylim, int):
        plt.ylim([0, ylim])
    elif ylim is not None:
        plt.ylim(list(ylim))


@torch.no_grad()
def draw(
        train_loss, train_accuracy, test_loss, test_accuracy, xlim=None, ylim_loss=None,
        ylim_accuracy=None
        ):
    plt.figure(figsize=(13, 4.8)).suptitle(f'Result')
    plt.subplot(121)
    _set_lim(xlim, ylim_loss)
    plt.plot(train_loss, color='blue', label='train loss')
    plt.plot(test_loss, color='red', label='test loss')
    plt.legend()
    plt.subplot(122)
    _set_lim(xlim, ylim_accuracy)
    plt.plot(train_accuracy, color='blue', label='train accuracy')
    plt.plot(test_accuracy, color='red', label='test accuracy')
    plt.legend()
    plt.show()


# 绘制混淆矩阵
@torch.no_grad()
def plot_confusion_matrix(
        cm: np.ndarray, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues
        ):
    """
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("显示百分比：")
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        print(cm)
    else:
        print('显示具体数字：')
        print(cm)
    plt.figure(figsize=(8, 8))  # 设置画布大小
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    # matplotlib版本问题，如果不加下面这行代码，则绘制的混淆矩阵上下只能显示一半，有的版本的matplotlib不需要下面的代码，分别试一下即可
    plt.ylim(len(classes) - 0.5, -0.5)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
                j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black"
                )
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# Roc曲线绘制
@torch.no_grad()
def ROC_AUC(label_data, pred_data, show_roc_curve=False):
    fpr, tpr, thresholds = m.roc_curve(label_data, pred_data)
    auc = m.auc(fpr, tpr)  # AUC其实就是ROC曲线下边的面积
    if show_roc_curve:
        plt.figure('ROC curve')
        plt.plot(fpr, tpr)
        plt.xlabel('fpr')
        plt.ylabel('tpr')
        plt.show()
    return fpr, tpr, auc


@torch.no_grad()
def PRC(label_data, output_data, show_prc_curve=False):
    # 返回查准率和召回率
    precision, recall, thresholds = m.precision_recall_curve(label_data, output_data)
    # 计算F1Score,它是用来均衡查准率和召回率的一个数据
    F1Score = 2 * (precision * recall) / (
            (precision + recall) + sys.float_info.min)
    # 找到F1Score对应的最大值的索引值
    MF = F1Score[np.argmax(F1Score)]
    # 计算平均精度
    AP = m.average_precision_score(label_data, output_data)
    # 输出PRC图象
    if show_prc_curve:
        plt.figure('Precision recall curve')
        plt.plot(recall, precision)
        plt.ylim([0.0, 1.0])
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.show()
    return recall, precision, MF, AP
