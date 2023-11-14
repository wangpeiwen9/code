# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2023/06/15
# Forum: www.medai.icu
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2023 All Rights Reserved.
from onekey_algo.segmentation.model_zoo.pranet.run_segmentation import main as seg_main
from onekey_algo import get_param_in_cwd

# 如果自己有coco格式的数据，可以直接使用自己的目录。
my_dir = get_param_in_cwd('data_dir')


# 设置参数
class params:
    dataset = r'vince_general_image_mask'
    data_path = my_dir
    model = 'PraNet'  # 'deeplabv3_resnet50'
    lr = 0.001
    workers = 0
    batch_size = 12
    decay_rate = 0.1
    decay_epoch = 50
    epochs = 100
    optimizer = 'sgd'
    momentum = 0.9
    weight_decay = 1e-4
    save_dir = get_param_in_cwd('save_dir', '.')
    resume = None
    save_per_epoch = True
    train_size = 352
    clip = 0.5
    attr = {}

    def __setattr__(self, key, value):
        self.attr[key] = value


# 训练模型
seg_main(params)
