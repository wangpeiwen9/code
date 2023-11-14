# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2022/3/22
# Home: www.medai.icu
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2022 All Rights Reserved.
import numpy as np

from onekey_algo.custom.utils.around import connect, fill_cross


def test_connect():
    data = np.array(
        [[1, 0, 0, 0, 0],
         [0, 0, 2, 1, 0],
         [0, 2, 3, 0, 0],
         [0, 1, 2, 1, 0],
         [0, 0, 0, 0, 1]])
    print(connect(data, 1, stride=5, mode='LINE'))
    print(fill_cross(data, 1))


if __name__ == '__main__':
    test_connect()
