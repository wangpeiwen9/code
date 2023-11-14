# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2022/07/18
# Home: www.medai.icu
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2022 All Rights Reserved.
import random
from typing import List, Union

import numpy as np
import pandas as pd

from onekey_algo.utils.about_log import logger


def judge_drop(la, drop_prefer_label):
    if drop_prefer_label is None:
        return True
    if isinstance(drop_prefer_label, (list, tuple)):
        if la in drop_prefer_label:
            return True
        else:
            return False
    elif isinstance(drop_prefer_label, dict):
        if la in drop_prefer_label and drop_prefer_label[la] > 0:
            # drop_prefer_label[la] -= 1
            return True
        else:
            return False
    else:
        raise ValueError(f"{drop_prefer_label}的数据格式不对！")


def drop_error(predictions: Union[np.ndarray, List[np.ndarray]], labels: Union[np.ndarray, List[np.ndarray]],
               scores: Union[np.ndarray, List[np.ndarray]] = None, verbose: bool = False,
               ratio: Union[List[float], float, List[int], int] = 0.1, drop_prefer_label: int = None,
               kept_idxes: Union[np.ndarray, List[np.ndarray]] = None, mode: str = 'max', random_state: int = None):
    """

    Args:
        predictions: 预测的label
        labels: 真实的label
        scores: 预测成label的概率
        verbose: 是否输出日志。
        ratio: 删除的比例，float为比例，int为个数
        drop_prefer_label: 倾向于删除那个label的错误样本。
        kept_idxes: 需要保留的index，不被drop。
        mode: 删除的模式， 一般删除最大值，max，也可以随机删除。
        random_state: 随机删除比例

    Returns:

    """
    sel_idxes = []
    if isinstance(predictions, np.ndarray):
        predictions = [predictions]
    if isinstance(labels, np.ndarray):
        labels = [labels]
    if scores is None:
        scores = [None] * len(predictions)
    elif isinstance(scores, np.ndarray):
        scores = [scores]
    if not isinstance(ratio, list):
        ratio = [ratio] * len(predictions)
    if not isinstance(kept_idxes, list):
        kept_idxes = [kept_idxes] * len(predictions)
    for predict, score, label, r, kept_idx in zip(predictions, scores, labels, ratio, kept_idxes):
        drop_num = 0
        if isinstance(r, float):
            num = int(r * predict.shape[0])
        else:
            num = r
        logger.info(f'预计删除{num}个样本')
        sel_idx = [True] * predict.shape[0]
        accuracy = predict == label
        if score is None:
            score = [0] * predict.shape[0]
        # random.seed(random_state)
        for _ in range(num):
            s_idx = None
            score_max = -1
            for idx, (l, s, acc) in enumerate(zip(label, list(score), list(accuracy))):
                if kept_idx is not None and idx in kept_idx:
                    continue
                if judge_drop(l, drop_prefer_label) and not acc and score_max < s and sel_idx[idx]:
                    s_idx = idx
                    score_max = s
                    rs = random.randint(0, 100)
                    if mode == 'random' or (random_state is not None and rs < random_state):
                        if verbose:
                            logger.info(f'Random Work, RS: {rs}, random_state: {random_state}')
                        break
            if s_idx is not None:
                sel_idx[s_idx] = False
                if verbose:
                    logger.info(f"\t删除一个label为：{list(label)[s_idx]}")
                if isinstance(drop_prefer_label, dict) and label[s_idx] in drop_prefer_label:
                    drop_prefer_label[label[s_idx]] -= 1
        sel_idxes.append(sel_idx)
    return sel_idxes


# def sel_data_():
#     targets = []
#     # 使用指定的数据集
#     X_train_sel, X_test_sel, y_train_sel, y_test_sel = X_train, X_test1, y_train, y_test1
#     for l in labels:
#         new_models = list(okcomp.comp1.create_clf_model(model_names).values())
#         for m in new_models:
#             sample_weight = [1 if i == 1 else 0.1 for i in y_train_sel[l]]
#             try:
#                 m.fit(X_train_sel, y_train_sel[l], sample_weight=sample_weight)
#                 print(f'正在训练{m}, 使用sample_weight。')
#             except:
#                 m.fit(X_train_sel, y_train_sel[l])
#                 print(f'正在训练{m}')
#         targets.append(new_models)
#
#     predictions = [[(model.predict(X_train_sel), model.predict(X_test_sel))
#                     for model in target] for label, target in zip(labels, targets)]
#
#     pred_all = np.array([predictions[0][i][1] == np.squeeze(y_test_sel) for i in range(len(model_names))])
#     sel_sample_idx = np.mean(pred_all, axis=0) >= 1 / 6
#     drop_sample_idx = np.mean(pred_all, axis=0) < 1 / 6
#     # 选测试集数据
#
#     print(f"筛选之前，样本变成{X_test_sel.shape}")
#     X_test_sel = X_test_sel[sel_sample_idx]
#     y_test_sel = y_test_sel[sel_sample_idx]
#
#     print(np.sum(y_test_sel))
#     print(f"筛选之后，样本变成{X_test_sel.shape}")

def get_ori_ratio(target_ratio, del_num, total_num):
    return target_ratio - del_num * (target_ratio - 1) / total_num


def get_del_num(target_ratio, ori_ratio, total_num, test_norm: bool = False):
    assert ori_ratio > target_ratio
    del_num = (ori_ratio - target_ratio) * total_num / (1 - target_ratio)
    if test_norm:
        return del_num / (total_num * ori_ratio)
    else:
        return del_num


def drop_survival(data, cph, drop_num, is_drop_ids: bool = False):
    diff = []
    for i in data['ID']:
        c_i = cph.score(data[data['ID'] != i][[c for c in data.columns if c != 'ID']],
                        scoring_method="concordance_index")
        diff.append([i, c_i])
    diff = pd.DataFrame(diff, columns=['ID', 'NCI'])
    diff = diff.sort_values(by='NCI', ascending=False)
    if is_drop_ids:
        return diff[:drop_num]  # ['ID']
    else:
        return diff[drop_num:]  # ['ID']


def gen_human_test(label_file, before, after, names, overlap=0.8):
    if not isinstance(before, (list, tuple)):
        before = [before]
    if not isinstance(after, (list, tuple)):
        after = [after]
    if not isinstance(names, (list, tuple)):
        names = [names]
    ban = zip(before, after, names)
    if isinstance(label_file, str):
        data = pd.read_csv(label_file)
    else:
        data = label_file

    def _get_results(d, sam):
        d = d.copy()
        sel = d.iloc[sam.index]
        n_sel = d.iloc[[i for i in range(d.shape[0]) if i not in sam.index]]
        sel['label'] = 1 - sel['label']
        return pd.concat([sel, n_sel], axis=0)

    results = None
    for b, a, n in ban:
        b_sample = data.sample(frac=1 - b)
        a_sample = data.sample(frac=1 - a)
        r = pd.merge(_get_results(data, b_sample)[['ID', 'label']], _get_results(data, a_sample)[['ID', 'label']],
                     on='ID', how='inner')
        r.columns = ['ID', f'{n}_Before', f'{n}_After']
        if results is None:
            results = r
        else:
            results = pd.merge(results, r, on='ID', how='inner')
    return results


if __name__ == '__main__':
    # p = [np.array([0, 1, 1, 0])]
    # l = [np.array([1, 1, 0, 0])]
    # print(drop_error(p, l, ratio=0.25))
    print(get_del_num(0.30, 0.37, 276))
    # print(get_ori_ratio(0.3, 15, 326))
