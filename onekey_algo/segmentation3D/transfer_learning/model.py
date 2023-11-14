# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2021/3/21
# Home: www.medai.icu
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2021 All Rights Reserved.

import torch
from torch import nn

from onekey_algo.utils.about_log import logger
from onekey_core.models import resnet3d as resnet


def generate_model(opt):
    assert opt.model in [
        'resnet'
    ]

    if opt.model == 'resnet':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]

        if opt.model_depth == 10:
            model = resnet.resnet10_3d(sample_input_W=opt.input_W, sample_input_H=opt.input_H,
                                       sample_input_D=opt.input_D,
                                       shortcut_type=opt.resnet_shortcut, no_cuda=opt.no_cuda,
                                       num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 18:
            model = resnet.resnet18_3d(sample_input_W=opt.input_W, sample_input_H=opt.input_H,
                                       sample_input_D=opt.input_D,
                                       shortcut_type=opt.resnet_shortcut, no_cuda=opt.no_cuda,
                                       num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 34:
            model = resnet.resnet34_3d(sample_input_W=opt.input_W, sample_input_H=opt.input_H,
                                       sample_input_D=opt.input_D,
                                       shortcut_type=opt.resnet_shortcut, no_cuda=opt.no_cuda,
                                       num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 50:
            model = resnet.resnet50_3d(sample_input_W=opt.input_W, sample_input_H=opt.input_H,
                                       sample_input_D=opt.input_D,
                                       shortcut_type=opt.resnet_shortcut, no_cuda=opt.no_cuda,
                                       num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 101:
            model = resnet.resnet101_3d(sample_input_W=opt.input_W, sample_input_H=opt.input_H,
                                        sample_input_D=opt.input_D,
                                        shortcut_type=opt.resnet_shortcut, no_cuda=opt.no_cuda,
                                        num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 152:
            model = resnet.resnet152_3d(sample_input_W=opt.input_W, sample_input_H=opt.input_H,
                                        sample_input_D=opt.input_D,
                                        shortcut_type=opt.resnet_shortcut, no_cuda=opt.no_cuda,
                                        num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 200:
            model = resnet.resnet200_3d(sample_input_W=opt.input_W, sample_input_H=opt.input_H,
                                        sample_input_D=opt.input_D,
                                        shortcut_type=opt.resnet_shortcut, no_cuda=opt.no_cuda,
                                        num_seg_classes=opt.n_seg_classes)
        else:
            raise ValueError(f"model_depth configure error {opt.model_depth}")
    else:
        raise ValueError(f"{opt.model_depth}不支持，只支持ResNet")
    if not opt.no_cuda:
        if len(opt.gpu_id) > 1:
            model = model.cuda()
            model = nn.DataParallel(model, device_ids=opt.gpu_id)
            net_dict = model.state_dict()
        else:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id[0])
            model = model.cuda()
            model = nn.DataParallel(model, device_ids=None)
            net_dict = model.state_dict()
    else:
        net_dict = model.state_dict()

    # load pretrain
    if opt.phase != 'test' and opt.pretrain_path:
        logger.info('loading pretrained model {}'.format(opt.pretrain_path))
        pretrain = torch.load(opt.pretrain_path)
        pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}

        net_dict.update(pretrain_dict)
        model.load_state_dict(net_dict)

        new_parameters = []
        for pname, p in model.named_parameters():
            for layer_name in opt.new_layer_names:
                if pname.find(layer_name) >= 0:
                    new_parameters.append(p)
                    break

        new_parameters_id = list(map(id, new_parameters))
        base_parameters = list(filter(lambda p: id(p) not in new_parameters_id, model.parameters()))
        parameters = {'base_parameters': base_parameters,
                      'new_parameters': new_parameters}

        return model, parameters

    return model, model.parameters()
