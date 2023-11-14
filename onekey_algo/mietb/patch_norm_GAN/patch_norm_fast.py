# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2023/08/10
# Home: www.medai.icu
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2023 All Rights Reserved.
# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2023/06/21
# Forum: www.medai.icu
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2023 All Rights Reserved.
import argparse
import os
import time
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from glob import glob
from onekey_algo.datasets.ClassificationDataset import ListDataset4Test
import slideflow as sf
from torchvision.transforms import transforms
from functools import partial

from onekey_algo.utils.MultiProcess import MultiProcess
from onekey_algo.utils.about_log import logger


def save_img(samples, thread_id, idir, save_dir, reference, method, batch_size, num_workers, verbose=1000):
    if thread_id != -1:
        logger.info(f"\t子任务{thread_id}: 获取{len(samples)}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    dataloader = torch.utils.data.DataLoader(ListDataset4Test(samples, trans, None),
                                             batch_size=batch_size, drop_last=False,
                                             shuffle=False, num_workers=num_workers)
    transform = sf.norm.autoselect(method, source=reference)
    done_num = 0
    start = time.time()
    for idx, (sample_, fnames) in enumerate(dataloader):
        if (idx + 1) % verbose == 0:
            logger.info(f'\t子任务：{thread_id}，速度： {(idx + 1) / (time.time() - start)} img/s')
        sample_ = sample_.to(device)
        output_data = transform.transform(sample_)
        for fname, pred in zip(fnames, output_data):
            rel2 = Path(fname).relative_to(Path(idir))
            try:
                os.makedirs(os.path.join(save_dir, os.path.dirname(rel2)), exist_ok=True)
                result_npy = pred.detach().cpu().numpy().transpose((1, 2, 0))
                result_png = Image.fromarray(result_npy)
                result_png.save(os.path.join(save_dir, rel2))
                done_num += 1
            except Exception as e:
                logger.error(f"\t{fname}遇到问题...{e}")
    return done_num


def patches_norm_dataloader(samples, idir, save_dir, reference='v3', batch_size=1, num_workers=1, num_process=1,
                            method='macenko'):
    save_img_sd = partial(save_img, idir=idir, save_dir=save_dir, method=method,
                          reference=reference, batch_size=batch_size, num_workers=num_workers)
    if num_process > 1:
        MultiProcess(samples, save_img_sd, num_process=num_process).run()
    else:
        save_img_sd(samples, thread_id=-1)


def main(input_dir, output_dir, method='macenko', num_workers=1, num_process: int = 1, batch_size=1,
         overwrite: bool = False):
    os.makedirs(output_dir, exist_ok=True)
    samples = []
    if os.path.exists(input_dir) and os.path.isdir(input_dir):
        for root, directory, filenames in os.walk(input_dir):
            logger.info(f"正在解析{root}...")
            sub_dir = root[root.find(input_dir) + len(input_dir):]
            sub_dir = sub_dir.strip('/').strip(r'\\')
            sub_dir_samples = [f for f in glob(os.path.join(root, '*.jpg')) + glob(os.path.join(root, '*.png'))
                               if '__MASK__.png' not in f]
            total_samples = len(sub_dir_samples)
            if not overwrite:
                sub_dir_samples = [s for s in sub_dir_samples
                                   if not os.path.exists(os.path.join(output_dir, sub_dir, os.path.basename(s)))]
            logger.info(f"\toverwrite: {overwrite}, 一共{total_samples}个样本，变成{len(sub_dir_samples)}。")
            samples.extend(sub_dir_samples)
        total_samples = len(samples)
        if total_samples == 0:
            logger.warning(f'{input_dir}没有找到任何样本。')
        else:
            logger.info(f"一共找到{total_samples}个样本")
            patches_norm_dataloader(samples, idir=input_dir, save_dir=output_dir, method=method,
                                    num_workers=num_workers, batch_size=batch_size, num_process=num_process)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('将所有的数据进行标准化！')
    parser.add_argument("--method", default='macenko', help="标准化方法，可以为rgb或者gray。")
    parser.add_argument("--overwrite", default=False, action='store_true', help="是否使用缓存机制")
    parser.add_argument("--num_process", default=4, type=int, help="整体进程的数据量")
    parser.add_argument("--num_workers", default=2, type=int, help="data loader的worker数量")
    parser.add_argument("--batch_size", default=1, type=int, help="batch size的大小")
    args = parser.parse_args()
    i_dir = r'F:\20230802-WX-MrZ\patches'
    o_dir = r'F:\20230802-WX-MrZ\patches_norm'
    for i_dir, o_dir in [  # (r'F:\20230802-WX-MrZ\patches', r'F:\20230802-WX-MrZ\patches_norm'),
        (r'F:\20230904-LiuHui\patches', r'F:\20230904-LiuHui\patches_norm')]:
        main(i_dir, o_dir, method=args.method, overwrite=args.overwrite,
             num_process=args.num_process, num_workers=args.num_workers, batch_size=args.batch_size)
    # input('按任意键退出...')
