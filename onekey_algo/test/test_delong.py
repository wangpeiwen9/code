# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2022/3/7
# Home: www.medai.icu
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2022 All Rights Reserved.
import random

import numpy as np

from onekey_algo.custom.components.delong import delong_roc_variance, delong_roc_test, calc_95_CI


def get_fake_data(num=100):
    gt = []
    pred1 = []
    pred2 = []
    for _ in range(num):
        gt.append(random.randint(0, 1))
        pred1.append(random.randint(0, 101) / 100)
        pred2.append(random.randint(0, 101) / 100)
    return np.array(gt), np.array(pred1), np.array(pred2)


def test_delong_roc_variance():
    gt, pred1, pred2 = get_fake_data()
    aucs, delongcov = delong_roc_variance(gt, pred1)
    print(aucs, delongcov)


def test_delong_roc_test():
    gt, pred1, pred2 = get_fake_data()
    p_value = delong_roc_test(gt, pred1, pred2)
    print(p_value)


def test_calc_95_CI():
    alpha = .95
    y_pred = np.array([0.21, 0.32, 0.63, 0.35, 0.92, 0.79, 0.82, 0.99, 0.04])
    y_true = np.array([0, 1, 0, 0, 1, 1, 0, 1, 0])

    auc, ci = calc_95_CI(y_true, y_pred, alpha=alpha, with_auc=True)
    print('AUC:', auc)
    print('95% AUC CI:', ci)


if __name__ == '__main__':
    test_delong_roc_test()
