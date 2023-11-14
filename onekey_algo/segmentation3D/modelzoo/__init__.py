# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2021/12/19
# Home: www.medai.icu
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2021 All Rights Reserved.
from monai.networks.layers import Norm

DEFAULT_MODEL_PARAMS = {'UNET': {'spatial_dims': 3,
                                 'in_channels': 1,
                                 'channels': (16, 32, 64, 128, 256),
                                 'strides': (2, 2, 2, 2),
                                 'num_res_units': 2,
                                 'norm': Norm.BATCH, },
                        'SEGRESNET': {'blocks_down': [1, 2, 2, 4],
                                      'blocks_up': [1, 1, 1],
                                      'init_filters': 16,
                                      'in_channels': 1,
                                      'dropout_prob': 0.2, },
                        'UNETR': {'in_channels': 1,
                                  'feature_size': 16,
                                  'hidden_size': 768,
                                  'mlp_dim': 3072,
                                  'num_heads': 12,
                                  'pos_embed': "perceptron",
                                  'norm_name': "instance",
                                  'res_block': True,
                                  'dropout_rate': 0.0, },
                        'VNET': {'spatial_dims': 3,
                                 'in_channels': 1, }}
