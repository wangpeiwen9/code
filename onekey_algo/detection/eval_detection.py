# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2021/5/11
# Home: www.medai.icu
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2021 All Rights Reserved.
import argparse
import json
import os
from typing import List, Tuple

import imgviz
import numpy as np
import torch
import torch.utils.data
from PIL import Image

from onekey_core.core import create_model
from onekey_core.transforms import transforms
from onekey_algo.utils import create_dir_if_not_exists


def inference(image_list, model, class_names, device, save_dir, score_thres=.5) -> List:
    create_dir_if_not_exists(save_dir)
    model.eval()
    if not isinstance(image_list, (list, tuple)):
        image_list = [image_list]
    transformer = transforms.Compose([transforms.ToTensor()])
    results = []
    with torch.no_grad():
        for image_path in image_list:
            img = np.array(Image.open(image_path).convert('RGB'))
            sample_ = transformer(img)
            sample_ = sample_.to(device)
            # print(sample_.size())
            outputs, = model(sample_.view(1, *sample_.size()))
            scores = outputs['scores']
            bboxes = outputs['boxes'][scores > score_thres]
            trans_bboxes = [[y1, x1, y2, x2] for x1, y1, x2, y2 in bboxes]

            labels = outputs['labels'][scores > score_thres]
            captions = [class_names[l] for l in labels]
            bboxviz = imgviz.instances2rgb(image=img, bboxes=trans_bboxes, labels=labels, captions=captions)
            imgviz.io.imsave(os.path.join(save_dir, os.path.basename(image_path)), bboxviz)
            results.append((os.path.basename(image_path),
                            os.path.join(save_dir, os.path.basename(image_path)),
                            list(zip(trans_bboxes, captions))))
    return results


def init(config_path):
    config = json.loads(open(os.path.join(config_path, 'task.json')).read())
    class_names = [l.strip() for l in open(os.path.join(config_path, 'labels.txt')).readlines()]
    model_path = os.path.join(config_path, 'BEST-training-params.pth')
    assert 'model_name' in config and 'num_classes' in config
    num_classes = len(class_names)

    # get the core using our helper function
    model = create_model(f"detection.{config['model_name']}", pretrained=False, num_classes=num_classes)
    device_info = 'cpu'
    device = torch.device(device_info)
    model = model.to(device)
    model.load_state_dict(state_dict=torch.load(model_path, map_location=device)['model'])
    model.eval()
    return model, class_names, device


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Segmentation Evaluating.')

    parser.add_argument('--data', default=None, nargs='*', required=True, help='dataset')
    parser.add_argument('--model_root', default=None, required=True, help='model_root')
    parser.add_argument('--device', default=None, choices=['cuda', 'cpu'], help='device')
    parser.add_argument('--save_dir', default='.', help='path where to save')
    args = parser.parse_args()

    model, class_names, device = init(args.model_root)
    # Inference
    inference(args.data, model, device, class_names=class_names, save_dir=args.save_dir)
