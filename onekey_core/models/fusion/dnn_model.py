# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2021/12/22
# Home: www.medai.icu
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2021 All Rights Reserved.
import os
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn

from onekey_algo.utils.about_log import logger

__all__ = ['CostumeNet']
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class BaseDNNNet(nn.Module):
    def __init__(self, input_dim: int, hidden_unit: List[int], dropout: Union[List[float], float] = None):
        super(BaseDNNNet, self).__init__()
        if dropout is None:
            dropout = [0] * len(hidden_unit)
        elif isinstance(dropout, (int, float)):
            dropout = [dropout] * len(hidden_unit)
        assert len(hidden_unit) == len(dropout)
        self.input_dim = input_dim
        self.hidden_unit = hidden_unit
        self.f_hidden_unit = [input_dim] + hidden_unit
        self.dropout = dropout
        dnn_units = []
        for former_unit, this_unit, dropout_prob in zip(self.f_hidden_unit, self.hidden_unit, self.dropout):
            dnn_unit = nn.Sequential(nn.Dropout(dropout_prob),
                                     nn.Linear(former_unit, this_unit),
                                     nn.ReLU(inplace=True))
            dnn_units.append(dnn_unit)
        self.layers = nn.Sequential(*dnn_units)

    def forward(self, x):
        return self.layers(x)

    @property
    def last_unit(self):
        return self.hidden_unit[-1]


def _check_drop(settings):
    dropout_settings = settings.get('dropout', None)
    hidden_len = len(settings['hidden_unit'])
    if dropout_settings is None:
        settings['dropout'] = [0] * hidden_len
    elif isinstance(dropout_settings, (list, tuple)):
        dropout_settings = list(dropout_settings)
        if dropout_settings[0] > 0:
            logger.warning(f'Dropout at input {dropout_settings[0]} may damage accuracy. We set to 0!')
            dropout_settings[0] = 0
            settings['dropout'] = dropout_settings
    elif isinstance(dropout_settings, (int, float)):
        if dropout_settings > 0:
            logger.warning(f'Dropout at input {dropout_settings} may damage accuracy. We set to 0!')
        settings['dropout'] = [0] + [dropout_settings] * (hidden_len - 1)
    else:
        raise ValueError(f"dropout type{type(dropout_settings)} not found!")
    assert all(0 <= s < 1 for s in settings['dropout'])
    return settings


class CostumeNet(nn.Module):
    def __init__(self, *inputs, kwarg_inputs=None, output_unit=2):
        super(CostumeNet, self).__init__()
        self.output_unit = output_unit
        if len(inputs) and kwarg_inputs is not None:
            logger.warning(f'inputs and kwargs_input can not be set simultaneously, use `inputs`!')
        if len(inputs):
            self.layers = nn.ModuleList()
            for i in inputs:
                self.layers.append(BaseDNNNet(**_check_drop(i)))
            total_features = sum([l.last_unit for l in self.layers])
        else:
            self.layers = nn.ModuleDict()
            for k, v in kwarg_inputs.items():
                self.layers[k] = BaseDNNNet(**_check_drop(v))
            total_features = sum([v.last_unit for v in self.layers.values()])
        self.task_spec_layer = nn.Linear(total_features, output_unit)

    def forward(self, x):
        if isinstance(self.layers, nn.ModuleList):
            assert isinstance(x, list) and len(x) == len(self.layers)
            outputs = [layer(data) for data, layer in zip(x, self.layers)]
        else:
            assert isinstance(x, dict)
            outputs = []
            for k, v in self.layers.items():
                outputs.append(v(x[k]))

        concat_features = torch.cat(outputs, dim=1)
        y = self.task_spec_layer(concat_features)
        return y


if __name__ == '__main__':
    # base_model = BaseDNNNet(10, [100, 1000, 20], [0.5] * 3)
    # print(base_model)

    config = {'input_dim': 10, 'hidden_unit': [32, 64, 128, 32], 'dropout': 0.5}
    cosnet = CostumeNet(kwarg_inputs={'in1': config, 'in2': config})
    print(cosnet({'in1': torch.tensor(np.ones([1, 10]), dtype=torch.float32),
                  'in2': torch.tensor(np.zeros([1, 10]), dtype=torch.float32)}))
    print(cosnet)
