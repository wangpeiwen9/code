# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2021/3/23
# Home: www.medai.icu
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2021 All Rights Reserved.
from onekey_algo.utils.ImageSpider import download_image

if __name__ == '__main__':
    download_image('labels.txt', engines=['sogou'], save_to='train', ask_num=100, num_process=4)
    download_image('labels.txt', engines=['bing'], save_to='val', ask_num=10, num_process=1)
