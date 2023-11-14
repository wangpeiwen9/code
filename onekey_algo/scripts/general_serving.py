# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2023/03/29
# Forum: www.medai.icu
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2023 All Rights Reserved.
import json
import os
import tempfile
from typing import Union

import joblib
import numpy as np
import pandas as pd
import yaml
from flask import request, Flask, jsonify

from onekey_algo.custom.components.Radiology import ConventionalRadiomics
from onekey_algo.custom.components.ml import inference as ml_inference
from onekey_algo.custom.components.ml import transform as ml_transform
from onekey_algo.segmentation3D.modelzoo.eval_3dsegmentation import inference as inference_seg3d
from onekey_algo.segmentation3D.modelzoo.eval_3dsegmentation import init as init_seg3d
from onekey_algo.utils.about_log import logger

app = Flask(__name__, static_folder='website/dist', static_url_path='/')
with open('config.yaml', encoding='utf-8') as f:
    config = yaml.load(f.read(), Loader=yaml.FullLoader)


class MLModel:
    def __init__(self, model_path, transform, sel_features=None):
        self.model = joblib.load(model_path)
        self.transform = transform
        self.sel_features = sel_features

    def predict(self, sample, **kwargs):
        for k in sample:
            sample[k] = ml_transform(sample[k], self.transform, sel_featues=self.sel_features)
        results = ml_inference(self.model, sample, **kwargs)
        return results


class SegModel(object):
    def __init__(self, model_name, model_path, roi_size=(48, 48, 48), **model_params):
        self.model, self.transform, self.device = init_seg3d(model_name, model_path, **model_params)
        self.roi_size = roi_size

    def predict(self, sample, save_dir=None, save_name='mask.nii.gz'):
        inference_seg3d(sample, self.model, self.transform, self.device, roi_size=self.roi_size,
                        save_dir=save_dir, save_name=save_name)


rad_model: Union[MLModel, None] = None
seg_model: Union[SegModel, None] = None
radio_exec: Union[ConventionalRadiomics, None] = None


def init_models():
    global rad_model, seg_model, radio_exec
    if 'rad_model' in config:
        rad_config = config['rad_model']
        rad_model = MLModel(rad_config['model_path'], rad_config['transform'], rad_config['sel_features'])
        radio_exec = ConventionalRadiomics(param_file=rad_config['feature_extractor'], correctMask=True)
        logger.info(f'成功加载影像组学模型...')
    if 'segmentation3d_model' in config:
        segmentation3d_config = config['segmentation3d_model']
        seg_model = SegModel(segmentation3d_config['model_name'],
                             segmentation3d_config['model_path'],
                             **segmentation3d_config['model_params'])
        logger.info(f'成功加载自动分割模型...')


@app.route('/resource', methods=['GET'])
def resource():
    return app.send_static_file(request.args.get('name'))


@app.route('/', methods=['GET'])
def home_page():
    return app.send_static_file('index.html')


@app.route('/api/config', methods=['GET'])
def get_serving_config():
    return jsonify(config)


def predict(image, mask=None, addition_features=None):
    global rad_model, seg_model, radio_exec
    radio_exec.extract([image], [mask])
    rad_features = radio_exec.get_label_data_frame()
    if addition_features is not None:
        feature_names, feature_values = zip(*addition_features)
        clinic_features = pd.DataFrame([[os.path.basename(image)] + list(feature_values)],
                                       columns=['ID'] + list(feature_names))
        rad_features = pd.merge(rad_features, clinic_features, on='ID', how='inner')
    # print(rad_features)
    results = rad_model.predict({os.path.basename(image): rad_features},
                                features=np.array(rad_features)[0][1:].tolist())
    return results


@app.route('/api/predict', methods=['POST'])
def submit():
    global rad_model, seg_model, radio_exec
    age = int(request.form.get('age'))
    sex = request.form.get('sex')
    # floatTest = request.form['floatTest']
    # intTest = request.form['intTest']
    image = request.files.get('image')
    mask = request.files.get('mask')
    with tempfile.TemporaryDirectory() as tmp_dir:
        # tmp_dir = r'C:\Users\onekey\Desktop\1'
        image_path = os.path.join(tmp_dir, image.filename)
        image.save(image_path)
        if mask is None:
            assert seg_model is not None
            mask_filename = 'mask.nii.gz'
            seg_model.predict(image_path, save_dir=tmp_dir, save_name=mask_filename)
            mask_path = os.path.join(tmp_dir, mask_filename)
        else:
            mask_path = os.path.join(tmp_dir, mask.filename)
            mask.save(mask_path)
        results = predict(image_path, mask_path, [('Age', age)])
        # print(results)
        json.dumps(results, ensure_ascii=False, indent=True)
        return json.dumps(results, ensure_ascii=False, indent=True)


if __name__ == '__main__':
    init_models()
    ipath = r'C:\models\images\1.nii.gz'
    mpath = r'C:\models\masks\1.nii.gz'
    # seg_model.predict(ipath, save_dir=r'C:\models', save_name='tmp.nii.gz')
    # print(predict(ipath, mpath, [('Age', 50)]))

    app.run('0.0.0.0', port=5000, debug=True)
