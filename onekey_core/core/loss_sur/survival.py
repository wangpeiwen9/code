# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2023/10/15
# Home: www.medai.icu
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2023 All Rights Reserved.
import torch


def cox_loss(y_true, y_pred):
    time_value = y_true[:, 0]
    event = y_true[:, 1].bool()
    score = y_pred.squeeze()

    ix = torch.where(event)

    sel_mat = (time_value[ix] <= time_value.view(-1, 1)).float().T
    p_lik = score[ix] - torch.log(torch.sum(sel_mat * torch.exp(score), dim=-1))

    loss = -torch.mean(p_lik)

    return loss


def concordance_index(y_true, y_pred):
    time_value = y_true[:, 0]
    event = y_true[:, 1].bool()

    # find index pairs (i,j) satisfying time_value[i]<time_value[j] and event[i]==1
    ix = torch.where((time_value.view(-1, 1) < time_value) & event.view(-1, 1))

    # count how many score[i]<score[j]
    s1 = y_pred[ix[0]]
    s2 = y_pred[ix[1]]
    ci = torch.mean((s1 < s2).float())

    return ci
