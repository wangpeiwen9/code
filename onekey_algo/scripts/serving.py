# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2021/10/17
# Home: www.medai.icu
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2021 All Rights Reserved.
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import traceback
import webbrowser

from flask import Flask, make_response, request, Response, jsonify

from onekey_algo.scripts.TaskViz import ClassificationVisualize, DetectionVisualize, SegmentationVisualize
from onekey_algo.utils import create_dir_if_not_exists

app = Flask(__name__, static_folder='html')

what_tasks: ClassificationVisualize = None
where_tasks: DetectionVisualize = None
which_tasks: SegmentationVisualize = None
IMAGE_STATIC = 'static'
create_dir_if_not_exists(IMAGE_STATIC)


@app.route('/', methods=['GET'])
def home_page():
    return app.send_static_file('pic.html')


@app.route('/model/test', methods=['POST'])
def predict():
    try:
        global what_tasks, where_tasks, which_tasks
        model_name = request.form.get('modelName', None)
        sample = request.files['file']
        if model_name:
            task_type, model_name = model_name.split('.')
            save_sample = os.path.join(IMAGE_STATIC, model_name, sample.filename)
            sample.save(save_sample)
            if task_type == 'what':
                results = what_tasks.predict(save_sample, model_name)
            elif task_type == 'where':
                results = where_tasks.predict(save_sample, model_name)
            elif task_type == 'which':
                results = which_tasks.predict(save_sample, model_name)
            else:
                return make_response('未发现的模型名称', 500)
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
        global what_tasks, where_tasks, which_tasks
        model_root = request.json['dir']
        if model_root and os.path.exists(model_root):
            what_tasks = ClassificationVisualize(model_root, save_dir=IMAGE_STATIC)
            resp.extend(what_tasks.viz())
            where_tasks = DetectionVisualize(model_root, save_dir=IMAGE_STATIC)
            resp.extend(where_tasks.viz())
            which_tasks = SegmentationVisualize(model_root, save_dir=IMAGE_STATIC)
            resp.extend(which_tasks.viz())
            return jsonify(resp)
        else:
            return make_response('目录不存在，请检查设置！', 500)
    except Exception as e:
        traceback.print_exc()
        return make_response(str(e), 500)


@app.route('/resource', methods=['GET'])
def resource():
    return app.send_static_file(request.args.get('name'))


@app.route(f'/{IMAGE_STATIC}/<model_name>/<image_name>')
def get_frame(model_name, image_name):
    # 图片上传保存的路径
    try:
        with open(os.path.join(IMAGE_STATIC, model_name, image_name), 'rb') as f:
            image = f.read()
            resp = Response(image, mimetype='image/svg+xml' if image_name.endswith('.svg') else 'image/png')
            return resp
    except Exception as e:
        return make_response(str(e), 500)


if __name__ == '__main__':
    webbrowser.open("http://127.0.0.1:8080/")
    app.run('127.0.0.1', port=8080, debug=False)
