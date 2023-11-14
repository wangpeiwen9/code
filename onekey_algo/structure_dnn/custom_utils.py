# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2021/12/24
# Home: www.medai.icu
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2021 All Rights Reserved.
import torch


def metric_func_clf(y_pred, y_true, onehot=False):
    if onehot:
        y_true = torch.argmax(y_true, dim=1)
    correct = torch.argmax(y_pred, dim=1) == y_true
    acc = torch.mean(correct.float())
    return float(acc)


def metric_func_reg(y_pred, y_true, **kwargs):
    mse = torch.mean(torch.pow(y_pred-y_true, 2))
    return float(mse)
