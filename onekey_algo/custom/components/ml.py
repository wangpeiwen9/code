# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2022/10/06
# Forum: www.medai.icu
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2022 All Rights Reserved.
import copy
import os

import joblib


def transform(samples, transform, sel_featues=None):
    new_samples = copy.deepcopy(samples)
    if sel_featues is not None:
        new_samples = new_samples[sel_featues]
    assert all(feat in new_samples.columns for feat in sel_featues), '所有的特征必须要全在数据集中！'
    for c in new_samples.columns:
        new_samples[c] = (new_samples[c] - transform[c]['mean']) / transform[c]['std']
    return new_samples


def inference(model, samples, output=None, **kwargs):
    if isinstance(model, str):
        assert os.path.exists(model), f"模型不存在：{model}"
        model = joblib.load(model)
    if isinstance(samples, (list, tuple)):
        samples = {idx: s for idx, s in enumerate(samples)}

    if isinstance(output, str):
        file = open(output, 'w', encoding='utf-8-sig')
    else:
        file = None

    results = {}
    for name, sample in samples.items():
        result = {'prediction': int(model.predict(sample)[0])}
        if 'Classifier' in str(type(model)):
            result['probably'] = model.predict_proba(sample)[0].tolist()
            result.update(kwargs)
        # print(f"{name}: {result}", file=file)
        results[name] = result
    return results
