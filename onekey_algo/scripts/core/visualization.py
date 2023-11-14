# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2021/08/12
# Home: www.medai.icu
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2021 All Rights Reserved.
import os
import os.path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc
from sklearn.metrics import roc_curve

from onekey_algo.utils import create_dir_if_not_exists
from onekey_algo.utils import parse_txt
from onekey_algo.utils.about_log import logger

matplotlib.pyplot.switch_backend('Agg')
__all__ = ['draw_roc', 'draw_confusion_matrix']


def draw_roc(onekey_result: str, save_dir: str = './'):
    """
    Draw roc and compute auc for binary classification
    Args:
        onekey_result: Onekey result file. filename, probability, prediction, gt
        save_dir: str, where to save. default ./

    Returns: auc, int
    """
    _, probability, pred, gt = zip(*parse_txt(onekey_result, convert2float_if_necessary=1))
    probability = [pro if pre == '1' else 1 - pro for pro, pre in zip(probability, pred)]
    assert len(set(gt)) == 2, 'Only binary classification task support roc drawing.'
    fpr, tpr, thresholds = roc_curve(list(map(lambda x: int(x), gt)), probability, pos_label=1)
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    create_dir_if_not_exists(save_dir)
    plt.savefig(os.path.join(save_dir, 'roc.svg'), dpi=300)
    return auc(fpr, tpr)


def draw_confusion_matrix(onekey_result: str, class_mapping: str = None, save_dir: str = './', num_classes: int = None):
    """

    Args:
        onekey_result: Onekey result file. filename, probability, prediction, gt
        class_mapping: mapping class index to readable classes.
        save_dir: str, where to save jpge file.
        num_classes: Number of classes.

    Returns:

    """
    _, _, prediction, gt = zip(*parse_txt(onekey_result, convert2float_if_necessary=1))
    num_classes = num_classes or len(set(gt))
    if num_classes != len(set(gt)):
        logger.warning(f'num_classes({num_classes}) is not equal to labels in gt({len(set(gt))}).')
    cm = np.zeros((num_classes, num_classes))
    for pred, y in zip(prediction, gt):
        cm[int(pred), int(y)] += 1
        cm[int(y), int(pred)] += 1

    mapping = {}
    if class_mapping and os.path.exists(class_mapping):
        label_names = (l_.strip() for l_ in open(class_mapping, encoding='utf8').readlines())
        mapping = dict(enumerate(label_names))

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # 按行进行归一化
    cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)
    str_cm = cm.astype(np.str).tolist()
    # 占比1%以下的单元格，设为0，防止在最后的颜色中体现出来
    # for i in range(cm.shape[0]):
    #     for j in range(cm.shape[1]):
    #         if int(cm[i, j] * 100 + 0.5) == 0:
    #             cm[i, j] = 0

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    # ax.figure.colorbar(im, ax=ax) # 侧边的颜色条带

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=map(lambda x: x if x not in mapping else mapping[x], range(num_classes)),
           yticklabels=map(lambda x: x if x not in mapping else mapping[x], range(num_classes)),
           title='混淆矩阵',
           ylabel='Actual',
           xlabel='Predicted')

    # 通过绘制格网，模拟每个单元格的边框
    ax.set_xticks(np.arange(cm.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 将x轴上的lables旋转45度
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # 标注百分比信息
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j] * 100 + 0.5) >= 0:
                ax.text(j, i, format(int(cm[i, j] * 100 + 0.5), fmt) + '%',
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    create_dir_if_not_exists(save_dir)
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.svg'), dpi=300)
