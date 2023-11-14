# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2019/7/25
# Home: www.medai.icu
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2019 All Rights Reserved.
import torch.nn as nn

from onekey_core.core import onekey_authorize
import torch

onekey_authorize.check_state()
__all__ = ['create_losses']


def cox_loss(y_true, y_pred):
    time_value = y_true[:, 0]
    event = y_true[:, 1].bool()
    score = y_pred.squeeze()

    ix = torch.where(event)

    sel_mat = (time_value[ix] <= time_value.view(-1, 1)).float().T
    print(sel_mat)
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


def create_losses(losses, **kwargs):
    r"""
    Create losses with specified loss name. Supported loss are as followings.
        'AdaptiveLogSoftmaxWithLoss', 'BCELoss', 'BCEWithLogitsLoss', 'CTCLoss', 'CosineEmbeddingLoss',
        'CrossEntropyLoss', 'HingeEmbeddingLoss', 'KLDivLoss', 'L1Loss', 'MSELoss', 'MarginRankingLoss',
        'MultiLabelMarginLoss', 'MultiLabelSoftMarginLoss', 'MultiMarginLoss', 'NLLLoss', 'NLLLoss2d',
        'PoissonNLLLoss', 'SmoothL1Loss', 'SoftMarginLoss', 'TripletMarginLoss'

    `losses` can be str for loss name or a list of dict type for multi losses combination.
        ```python
        loss1 = create_losses('ce', input=input, target=target)
        # Or a list of dict type.
        loss2 = create_losses([{'loss':'softmax_ce', kwargs:{'reduction':'mean'}},
            {"loss":'sigmoid', kwargs:{'reduction':'mean', 'pos_weight':None}}])
        ```
        The following is dict's params
        :param: loss, specify the loss. REQUIRED!
        :param: kwargs, other loss settings.

    :param losses: loss name or a list of dict type for multi losses combination.
    :param kwargs: other loss settings.
    :return: combined loss.
    :raises:
        ValueError, loss not found.
        AssertError
            type of each item in `losses` is not dict if use losses combination.
            `loss` not found in multi losses combination settings.

    """
    supported_losses = {'softmax_ce': nn.CrossEntropyLoss,  # Softmax cross entropy for single label.
                        'sigmoid_ce': nn.BCEWithLogitsLoss,  # Sigmoid cross entropy for single label.
                        'bce': nn.BCELoss,  # Binary classification targets without sigmoid activation.
                        'cosine_embedding': nn.CosineEmbeddingLoss,  # Cosine embedding loss.
                        'ctc': nn.CTCLoss,  # CTC loss.
                        'hinge': nn.HingeEmbeddingLoss,
                        'kl': nn.KLDivLoss,  # KL divergence Loss for continuous targets.
                        'l1': nn.L1Loss,
                        'smooth_l1': nn.L1Loss,
                        'triplet': nn.TripletMarginLoss,  # Triplet loss.
                        'mse': nn.MSELoss,
                        'ranking': nn.MarginRankingLoss,
                        'multi_sigmoid': nn.MultiLabelSoftMarginLoss,  # Multi label sigmoid loss.
                        'cox_loss': cox_loss
                        }

    def _form_loss(loss_name, **spec_loss_kwargs):
        if loss_name not in supported_losses:
            raise ValueError(f'Loss name {loss_name} not supported!')
        return supported_losses[loss_name](**spec_loss_kwargs)

    if isinstance(losses, list):
        assert all(isinstance(l, dict) and 'loss' in l for l in losses)
        for l in losses:
            if 'kwargs' not in l:
                l['kwargs'] = {}
        return [_form_loss(l['loss'], **l['kwargs']) for l in losses]
    else:
        return _form_loss(losses, **kwargs)
