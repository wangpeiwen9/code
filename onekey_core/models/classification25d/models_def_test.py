# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2023/07/21
# Home: www.medai.icu
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2023 All Rights Reserved.

import torch

from onekey_core.models.classification25d.resnet_fusion import resnet101


def test_resnet_fusion():
    img = torch.rand([8, 3, 3, 224, 224])
    cli = torch.rand([8, 32])
    model = resnet101(pretrained=True, slice_size=3, clinic_size=32, num_classes=2, e2e_comp=32)
    # print(model({'image': img, 'clinic': cli}))


if __name__ == '__main__':
    pass
