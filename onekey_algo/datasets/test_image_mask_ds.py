# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2021/5/28
# Home: www.medai.icu
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2021 All Rights Reserved.
import numpy as np

from onekey_algo.datasets.ImageMaskDataset import create_image_mask_dataset


def test_image_mask_dataset():
    ds = create_image_mask_dataset('test_case')
    for img, msk in ds:
        assert img.size == msk.size
        assert np.unique(np.array(msk)).shape == (2,)
