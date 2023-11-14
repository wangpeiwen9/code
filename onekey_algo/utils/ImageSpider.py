# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2021/3/21
# Home: www.medai.icu
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2021 All Rights Reserved.
import os
import random
import re
import threading
import time
import urllib.parse
import urllib.request
from typing import List

from PIL.ImageFile import ImageFile

from onekey_algo.utils.MultiProcess import MultiProcess
from onekey_algo.utils.about_log import logger
from onekey_algo.utils.common import create_dir_if_not_exists, check_directory_exists

ImageFile.LOAD_TRUNCATED_IMAGES = True


class DownloadImage(object):
    headers = {'User-Agent': "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) "
                             "Chrome/45.0.2454.101 Safari/537.36"}
    # 不同搜索引擎的图片url提取的正则表达式不一样
    img_regx = {
        'google': r'\\"ou\\":\\"(.*?)\\"',
        'bing': 'src="(.*?)"',
        'sogou': '"pic_url":"(.*?)"',
        'baidu': '"objURL":"(.*?)"',
        '360': '"img":"(.*?)"'
    }
    query_list = []

    def __init__(self, query_list, save_to):
        # setting proxy for google engine
        proxy_handler = urllib.request.ProxyHandler({'socks': '127.0.0.1'})
        opener = urllib.request.build_opener(proxy_handler)
        urllib.request.install_opener(opener)
        self.query_list = query_list
        self.save_to = save_to

    @staticmethod
    def get_engine_url(engine_name, query, bat_num):
        if engine_name == 'google':
            url = 'https://www.google.com/search'
            params = {
                'hl': 'en',
                'yv': 2,
                'tbm': 'isch',
                'q': query,
                'ijn': bat_num,
                'start': bat_num * 100,
                'asearch': 'ichunk',
                'async': '_id:rg_s,_pms:s'
            }
        elif engine_name == 'sogou':
            url = 'http://pic.sogou.com/pics'
            params = {
                'query': query,
                'mode': 1,
                'start': bat_num * 48,
                'reqType': 'ajax',
                'reqFrom': 'result',
                'tn': 0
            }
        elif engine_name == 'baidu':  # 注意百度的搜索词不需要进行编码
            url = "http://image.baidu.com/search/avatarjson"
            params = {
                'tn': 'resultjsonavatarnew',
                'ie': 'utf-8',
                'word': query,
                'rn': 60,
                'pn': bat_num * 60
            }
        elif engine_name == 'bing':
            url = 'http://www.bing.com/images/async'
            params = {
                'q': query,
                'first': bat_num * 36,
                'count': 35,
                'relp': 35,
                'lostate': 'r',
                'mmasync': 1
            }
        elif engine_name == '360':
            url = 'http://image.so.com/j'
            params = {
                'q': query,
                'src': 'srp',
                'correct': query,
                'sn': bat_num * 60,
                'pn': 60
            }
        else:
            raise ValueError('Search engine %s not found!' % engine_name)
        return url + "?" + urllib.parse.urlencode(params)

    def download_img(self, engine, query, batch, ask_num=2000):
        totalSum = 0
        for bat_num in range(0, batch):
            # 生成文件夹（如果不存在的话）
            path = os.path.join(self.save_to, query)
            create_dir_if_not_exists(path)
            try:
                url = DownloadImage.get_engine_url(engine, query, bat_num)
                req = urllib.request.Request(url, headers=self.headers)
                res = urllib.request.urlopen(req)
                page = res.read()
                # print(page)
                # 因为JSON的原因，在浏览器页面按F12看到的，和你打印出来的页面内容是不一样的，所以匹配的是objURL这个东西，
                # 对比一下页面里别的某某URL，那个能访问就用那个
                images = re.findall(self.img_regx[engine], str(page), re.S)
            except IOError as e:
                # 如果访问失败，就跳到下一个继续执行代码，而不终止程序
                logger.error('Query: %s encounter an error %s.' % (query, e))
                continue

            cur_img_id = 0
            # 访问上述得到的图片路径，保存到本地
            for src in images:
                src = src.replace('\\', '')
                try:
                    logger.info("Downloading No.%d from %s" % (cur_img_id + bat_num * len(images), src))
                    # 设置一个urlopen的超时，如果3秒访问不到，就跳到下一个地址，防止程序卡在一个地方。
                    img = urllib.request.urlopen(src, timeout=3)
                    img_content = img.read()
                    if img_content:
                        filename = "%d.jpg" % int(time.time() + random.random() * 1000)
                        with open(os.path.join(path, filename), 'wb') as img_file:
                            img_file.write(img_content)
                except Exception as e:
                    logger.info("%s, No.%d error: %s" % (query, cur_img_id + bat_num * len(images), e))
                    continue
                cur_img_id += 1
                totalSum += 1
                if totalSum > ask_num:
                    break
            if totalSum > ask_num:
                break


class ImageDownloadThread(threading.Thread):
    def __init__(self, engine, batch, img_download: DownloadImage, ask_num: int = 2000):
        threading.Thread.__init__(self)
        self.engine = engine
        self.query_list = img_download.query_list
        self.batch = batch
        self.img_download = img_download
        self.ask_num = ask_num

    def run(self):
        logger.info('starting download from engine:%s, batch=%d' % (self.engine, self.batch))
        for query in self.query_list:
            logger.info('start downloading images from engine: %s, query:%s' % (self.engine, query))
            # 以下是对已下载的图片进行resize操作
            # print('start resizing dir:%s' % query)
            # img_download.img_preprocess(query)
            self.img_download.download_img(self.engine, query, self.batch, ask_num=self.ask_num)
        logger.info('finished download batch %d from engine:%s' % (self.batch, self.engine))


def _download_image_single(samples, thread_id, save_to, engines, ask_num=2000):
    threads = []
    logger.info('THREAD_ID: %d start downloading...' % thread_id)
    img_download = DownloadImage(samples, save_to)
    for engine in engines:
        if engine == 'bing':
            batch = 20
        else:
            batch = 50
        download_thread = ImageDownloadThread(engine, batch, img_download, ask_num=ask_num)
        download_thread.start()
        threads.append(download_thread)

    for t in threads:  # join all threads for terminate
        t.join()


def download_image(data, save_to: str, engines: List[str] = None, ask_num: int = 2000, num_process: int = 1):
    """Download image form search engine('google', 'bing', 'baidu', 'sogou', '360'), you can assign `engines` for
    specific engine.

    :param data: Query data file, each line is a isolate query.
    :param save_to: Where to store image.
    :param engines: Search engines used.
    :param ask_num: How many examples asked per query.
    :param num_process: Number of process to be used, default 1.
    :return:
    """
    # Check settings.
    supported_engine = {'google', 'bing', 'baidu', 'sogou', '360'}
    if engines is None:
        engines = ['bing', 'baidu', 'sogou', '360']
    assert all([e in supported_engine for e in engines]), \
        'Search engine only %s supported' % ','.join(supported_engine)

    query_list = []
    if isinstance(data, str):
        check_directory_exists(data)
        with open(data, encoding='utf8') as f:
            for l in f.readlines():
                query_list.append(l.strip())
    elif isinstance(data, (list, tuple)):
        query_list = data
    else:
        raise ValueError(f'Only file path and list of category supported! {data} not supported')

    for q in query_list:
        create_dir_if_not_exists(os.path.join(save_to, q))
    logger.info('Staring...')
    MultiProcess(query_list, _download_image_single, num_process=num_process,
                 save_to=save_to, engines=engines, ask_num=ask_num).run(0)
