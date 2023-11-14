# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2023/06/13
# Forum: www.medai.icu
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2023 All Rights Reserved.
import copy


def param_dfs(kv_pairs, expand_param_kv):
    """
    Search possible parameters using DFS.

    :param kv_pairs: Former kv pairs.
    :param expand_param_kv: kv needs to expand.
    :return:
    """
    assert len(expand_param_kv) > 0, "`expand_param_kv` at least has one elements."
    k_, v_ = expand_param_kv.pop(0)
    new_pairs = []
    for param_v_ in v_:
        dc_kv_pairs = copy.deepcopy(kv_pairs)
        for kv_pair in dc_kv_pairs:
            kv_pair.append((k_, param_v_))
            new_pairs.append(kv_pair)
    if len(expand_param_kv) == 0:
        return new_pairs
    else:
        return param_dfs(new_pairs, expand_param_kv)


def construct_param_grid(model_config_):
    """
    Construct parameter grid if has any parameters start with auto__.

    :param model_config_: Model configurations, param correctness will not be checked!
    :return: Parameters grid.
    """
    static_params = {k_: v_ for k_, v_ in model_config_.items() if not k_.startswith('auto__')}
    dynamic_params = [(k_[6:], v_) for k_, v_ in model_config_.items() if k_.startswith('auto__')]
    if len(dynamic_params) == 0:
        return [model_config_]
    assert all(isinstance(v_, (list, tuple)) for _, v_ in dynamic_params), "All auto params must be list or tuple."
    dynamic_params = param_dfs([[]], dynamic_params)

    # Form param grid.
    param_grid = []
    for dynamic_param in dynamic_params:
        new_param = copy.deepcopy(static_params)
        new_param.update(dynamic_param)
        param_grid.append(new_param)
    return param_grid


if __name__ == '__main__':
    print(construct_param_grid({'lr': 1, 'auto__optimizer': ['sgd', 'adam']}))
