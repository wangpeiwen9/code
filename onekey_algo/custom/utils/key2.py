# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2022/7/4
# Home: www.medai.icu
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2022 All Rights Reserved.
import re
from typing import Union, List

import numpy as np
import pandas as pd
from onekey_algo.utils.about_log import logger
from sklearn.feature_extraction.text import TfidfVectorizer


def __convert2list(a):
    if not isinstance(a, list):
        return [a]
    else:
        return a


def removePunctuation(query):
    # 去除标点符号（只留字母、数字、中文)
    if query:
        rule = re.compile(u"[^a-zA-Z0-9\u4e00-\u9fa5| ]")
        query = rule.sub('', query)
    return query


def key2histogram(data: pd.DataFrame, group_column: str, histo_columns: Union[str, List[str]],
                  histo_lists: Union[list, List[list]] = None, default_value=0, norm: bool = False):
    """
    所有的数据生成直方图特征， 多个histo_columns存在是，所有的特征进行横向拼接。
    Args:
        data: 数据
        group_column: 样本分组的列明，一般为ID
        histo_columns: 用来计算直方图的列，如果为多列，则每列计算完直方图，然后特征拼接
        histo_lists: None或者与histo_columns个数相同，为自己指定特征列表
        default_value: 不存在特征时的默认值
        norm: 要不要归一化。
    Returns:

    """
    histo_columns = __convert2list(histo_columns)
    column_names = ['ID']
    if histo_lists is None:
        histo_lists = [None] * len(histo_columns)
    else:
        assert isinstance(histo_lists, list), f'histo_lists：{histo_lists}必须是list类型。'
        if not isinstance(histo_lists[0], list):
            histo_lists = [histo_lists]
    assert len(histo_columns) == len(histo_lists), f"长度必须相等！"
    assert all(c in data.columns for c in histo_columns + [group_column]), f"{histo_columns}或者" \
                                                                           f"{group_column}没有在数据中！"
    features = {}
    has_column_names = False
    groups = np.unique(np.array(data[group_column]))

    for idx, (histo_column, histo_list) in enumerate(zip(histo_columns, histo_lists)):
        if histo_list is None:
            histo_lists[idx] = sorted(list(set(data[histo_column])))
        else:
            assert len(histo_list) == len(set(histo_list)), 'histo_list不能包含重复元素'
    for idx, group in enumerate(groups):
        if len(groups) > 1000 and (idx + 1) % (len(groups) // 10) == 0:
            logger.info(f"处理完成{idx + 1}个，占比{(idx + 1) * 100 / len(groups):.2f}%")
        features[group] = []
        for histo_column, histo_list in zip(histo_columns, histo_lists):
            if not has_column_names:
                column_names.extend(list(map(lambda x: f"{histo_column}-{x}", histo_list)))
            feature = [default_value] * len(histo_list)
            for element in data[data[group_column] == group][histo_column]:
                feature[histo_list.index(element)] += 1
            if norm:
                feature = [f / sum(feature) for f in feature]
            features[group].extend(feature)
        has_column_names = True
    feature_array = [[k_] + v_ for k_, v_ in features.items()]
    return pd.DataFrame(feature_array, columns=column_names)


def key2tfidf(data: pd.DataFrame, group_column: str, corpus_columns: Union[str, List[str]]):
    """
    所有的数据生成直方图特征， 多个corpus_columns存在是，所有的特征进行横向拼接。
    Args:
        data: 数据
        group_column: 样本分组的列明，一般为ID
        corpus_columns: 用来计算作为语料的列明。
    Returns:

    """
    corpus_columns = __convert2list(corpus_columns)
    assert all(c in data.columns for c in corpus_columns + [group_column]), f"{corpus_columns}或者" \
                                                                            f"{group_column}没有在数据中！"
    features = {}
    tfidf_vec = TfidfVectorizer()
    groups = np.unique(np.array(data[group_column]))
    for group in groups:
        for corpus_column in corpus_columns:
            features[group] = (' '.join([f"{corpus_column}_{element}"
                                         for element in data[data[group_column] == group][corpus_column]]))
    ids = list(features.keys())
    corpus = [removePunctuation(v_) for v_ in features.values()]
    tfidf_matrix = tfidf_vec.fit_transform(corpus).toarray()
    feature_names = tfidf_vec.get_feature_names_out()
    tfidf_features = pd.DataFrame(tfidf_matrix, columns=feature_names, index=ids)
    tfidf_features['ID'] = tfidf_features.index
    return tfidf_features[['ID'] + list(c for c in tfidf_features.columns if c!='ID')]


if __name__ == '__main__':
    d = pd.DataFrame([['a', '1'],
                      ['a', '2'],
                      ['a', '1'],
                      ['a', '1'],
                      ['b', '1'],
                      ['b', '1'],
                      ['b', '1']],
                     columns=['name', 'prob'])
    print(key2histogram(d, 'name', 'prob'))
