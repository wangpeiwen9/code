# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2021/10/17
# Home: www.medai.icu
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2021 All Rights Reserved.
import os
import webbrowser

from onekey_algo.scripts.eye_task_viz import EyeTaskViz
from onekey_algo.utils.about_log import logger

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import traceback

from flask import Flask, make_response, request, Response, jsonify

from onekey_algo.utils import create_dir_if_not_exists

app = Flask(__name__, static_folder='html')

eye_tasks: EyeTaskViz

IMAGE_STATIC = 'static'
create_dir_if_not_exists(IMAGE_STATIC)


@app.route('/', methods=['GET'])
def home_page():
    return app.send_static_file('pic.html')


@app.route('/model/test', methods=['POST'])
def predict():
    try:
        global eye_tasks
        model_name = request.form.get('modelName', None)
        sample = request.files['file']
        if model_name:
            save_sample = os.path.join(IMAGE_STATIC, sample.filename)
            sample.save(save_sample)
            logger.info(f"模型名称：{model_name}, 样本：{save_sample}")
            results = eye_tasks.predict(save_sample, model_name)
            print(results)
            return jsonify(results)
        else:
            return make_response('目录不存在，请检查设置！', 500)
    except Exception as e:
        traceback.print_exc()
        return make_response(str(e), 500)


@app.route('/model/get', methods=['POST'])
def analysis():
    try:
        resp = []
        global eye_tasks
        model_root = request.json['dir']
        if model_root and os.path.exists(model_root):
            eye_tasks = EyeTaskViz(model_root, save_dir=IMAGE_STATIC)
            resp.extend(eye_tasks.viz_spec())
            return jsonify(resp)
        else:
            return make_response('目录不存在，请检查设置！', 500)
    except Exception as e:
        traceback.print_exc()
        return make_response(str(e), 500)


@app.route('/resource', methods=['GET'])
def resource():
    return app.send_static_file(request.args.get('name'))


@app.route(f'/{IMAGE_STATIC}/<task_type>/<model_name>/<image_name>')
def get_frame(task_type, model_name, image_name):
    # 图片上传保存的路径
    try:
        with open(os.path.join(IMAGE_STATIC, task_type, model_name, image_name), 'rb') as f:
            image = f.read()
            resp = Response(image, mimetype='image/svg+xml' if image_name.endswith('.svg') else 'image/png')
            return resp
    except Exception as e:
        return make_response(str(e), 500)


if __name__ == '__main__':
    webbrowser.open("http://127.0.0.1:8080/")
    app.run('127.0.0.1', port=8080, debug=False)
