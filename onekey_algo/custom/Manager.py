# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2022/3/16
# Home: www.medai.icu
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2022 All Rights Reserved.
import os
import shutil
from typing import List

import yaml

from onekey_algo import get_config

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import abc
import json
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.editor import ipython_display
from onekey_algo import get_param_in_cwd
from datetime import datetime
from onekey_algo.utils import check_directory_exists, check_file_exists
from onekey_algo.utils.about_log import logger


class ResourceManager(metaclass=abc.ABCMeta):
    def __init__(self, root=None):
        self.root = root or os.getenv('ONEKEY_VIDEO')
        assert self.root and os.path.exists(self.root) and os.path.isdir(self.root), f'资源目录不存在或者设置错误！'

    @abc.abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError

    @abc.abstractmethod
    def save_config(self):
        raise NotImplementedError


def cut_video(vp, s, e):
    logger.info(f"数据准备中，请稍候...")
    fname, ext = os.path.splitext(os.path.basename(vp))
    mp = os.path.join(os.path.dirname(vp), f"{fname}_{s}_{e}{ext}")
    if not os.path.exists(mp):
        clip = VideoFileClip(vp).subclip(s, e)
        clip.write_videofile(mp, logger=None)
    return mp


class VideoManager(ResourceManager):
    """
    root
      |- Video
        |- config.json
        |- dir1/comp1.mp4
        |- dir2/comp2.mp4
    """

    def __init__(self, root=None, config: str = 'config.json'):
        super().__init__(root)
        self.root = os.path.join(self.root, 'video')
        check_directory_exists(self.root, prefixes='视频目录')
        self.config_file = os.path.join(self.root, config)
        check_file_exists(self.config_file, force_file=True)
        with open(self.config_file, encoding='utf8') as cf:
            self.config = json.load(cf)

    def __getitem__(self, item):
        comp, *module = item.split('|')
        module = ''.join(module)
        if comp not in self.config:
            raise ValueError('你需要从baidu网盘下载更新OnekeyVideo资源'
                             '链接：https://pan.baidu.com/s/1x_FovKJYqcddtQSLTtuGFA?pwd=rb7b 密码：www.medai.icu')
        vp = os.path.join(self.root, self.config[comp]['__path__'])
        fname, ext = os.path.splitext(vp)
        if module:
            if '__path__' in self.config[comp][module] and os.path.exists(self.config[comp][module]['__path__']):
                return self.config[comp][module]['__path__']
            else:
                logger.info(f'数据准备中，请稍候...')
                s = self.config[comp][module]['start']
                e = self.config[comp][module]['end']
                clip = VideoFileClip(vp).subclip(s, e)
                mp = os.path.join(self.root, f"{fname}_{module}{ext}")
                clip.write_videofile(mp, logger=None)
                self.config[comp][module]['__path__'] = mp
                self.save_config()
                return mp
        else:
            return vp

    def save_config(self):
        with open(self.config_file, 'w', encoding='utf8') as cf:
            print(json.dumps(self.config, ensure_ascii=False, indent=True), file=cf)


def onekey_show(key, root=None, start=None, end=None, fmt="%H:%M:%S", width=980):
    disable_video = str(get_param_in_cwd('DISABLE_VIDEO', False))
    try:
        if os.getenv('DISABLE_VIDEO', 'False') == 'False' and disable_video == 'False':
            if os.path.exists(key):
                if start is None and end is None:
                    return ipython_display(key, maxduration=65535, width=width)
                elif start is not None and end is not None:
                    try:
                        start = datetime.strptime(start, fmt)
                        start = int((datetime.combine(datetime.min, start.time()) - datetime.min).total_seconds())
                        end = datetime.strptime(end, fmt)
                        end = int((datetime.combine(datetime.min, end.time()) - datetime.min).total_seconds())
                        assert start < end, '视频结束时间不能小于开始时间！'
                        return ipython_display(cut_video(key, s=start, e=end), maxduration=65535, width=width)
                    except Exception as e:
                        raise ValueError(f'获取视频片段错误：{e}')
            else:
                vm = VideoManager(root=root)
                return ipython_display(vm[key], maxduration=65535, width=width)
        else:
            logger.info("播放视频功能已经设置成：Disable！")
    except Exception as e:
        logger.error(f'获取视频失败，因为：{e}')


class SolManager(object):
    RUN_TEMP = r'jupyter nbconvert --ExecutePreprocessor.timeout=600 --to notebook --execute {notebook}'

    def __init__(self, cv_files: List[str], target: str, config_file='config.yaml', save_dir='.',
                 exe_order=None):
        assert all(os.path.exists(cf) for cf in cv_files)
        assert os.path.exists(target)
        self.target = target
        self.cv_files = cv_files
        self.config_file = config_file
        self.save_dir = save_dir
        self.cv_dir = []
        if exe_order is not None:
            self.exe_order = exe_order
        else:
            self.exe_order = sorted([f for f in os.listdir(target) if f.endswith('ipynb')])

    def cp_files(self):
        config = get_config(directory=self.target, config_file=self.config_file)
        for cf in self.cv_files:
            cf_name = os.path.splitext(os.path.basename(cf))[0]
            save2 = os.path.join(self.save_dir, cf_name)
            self.cv_dir.append(save2)
            shutil.copytree(self.target, save2)
            with open(os.path.join(save2, 'config.yaml'), 'w') as new_config_file:
                config['label_file'] = cf
                yaml.dump(config, new_config_file)

    def run(self):
        for word_dir in self.cv_dir:
            os.chdir(word_dir)
            for ipynb in self.exe_order:
                os.system(self.RUN_TEMP.format(notebook=ipynb))


if __name__ == '__main__':
    cv_files = [r'/Users/zhangzhiwei/Project/onekey-sol/TaskSpec55-SongXueFei/split_info/label-CV-0.csv']
    target = r'/Users/zhangzhiwei/Desktop/Project/onekey/onekey_comp/comp9-Solutions/sol3. 传统组学-多中心-瘤内瘤周-临床'
    save_dir = r'/Users/zhangzhiwei/Downloads/未命名文件夹'
    solm = SolManager(cv_files=cv_files, target=target, config_file='config.txt', save_dir=save_dir)
    solm.cp_files()
