# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2023/06/21
# Forum: www.medai.icu
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2023 All Rights Reserved.
import argparse
import os
from functools import partial

import cv2
import torch
import numpy as np
from glob import glob
from CasUNet import CasUNet
from onekey_algo.datasets.ClassificationDataset import ListDataset4Test
from torchvision.transforms import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def loader(src_path, method):
    if method == 0:
        src_data = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    elif method == 1:
        src_data = cv2.imread(src_path, cv2.IMREAD_COLOR)
        src_data = cv2.decolor(src_data)[0]
    else:
        raise Exception("invalid method")
    return src_data


def patches_norm(samples, net, save_dir='.', method=0):
    os.makedirs(save_dir, exist_ok=True)

    for idx, src_path in enumerate(samples):
        try:
            basename = os.path.basename(src_path)
            src_data = loader(src_path, method=method)
            src_data = (src_data / 127.5 - 1).astype(np.float32)
            input_data = torch.from_numpy(src_data[np.newaxis, np.newaxis, :]).to(device)
            output_data = net(input_data)

            result_npy = output_data.detach().cpu().numpy().transpose((0, 2, 3, 1))[0]
            result_png = np.uint8(np.clip((result_npy + 1) * 127.5, 0, 255))
            cv2.imwrite(os.path.join(save_dir, basename), result_png)
        except Exception as e:
            print(f'\t解析{src_path}遇到问题：{e}')


def patches_norm_dataloader(samples, net, save_dir, batch_size=1, num_workers=1, method=0):
    os.makedirs(save_dir, exist_ok=True)
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    this_loader = partial(loader, method=method)
    dataloader = torch.utils.data.DataLoader(ListDataset4Test(samples, trans, this_loader),
                                             batch_size=batch_size, drop_last=False,
                                             shuffle=False, num_workers=num_workers)
    for idx, (sample_, fnames) in enumerate(dataloader):
        try:
            sample_ = sample_.to(device)
            output_data = net(sample_)
            for fname, pred in zip(fnames, output_data):
                result_npy = pred.detach().cpu().numpy().transpose((1, 2, 0))
                result_png = np.uint8(np.clip((result_npy + 1) * 127.5, 0, 255))
                cv2.imwrite(os.path.join(save_dir, os.path.basename(fname)), result_png)
        except Exception as e:
            print(f"\t{fnames}存在处理错误{e}！")


def main(input_dir, output_dir, model_path=None, method=0, num_workers=1, batch_size=1, overwrite: bool = False):
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = model_path or os.path.join(os.environ.get('ONEKEY_HOME'), 'pretrain', f'patch_GAN.pth')
    net = CasUNet(1, 1, 3).to(device)
    net.load_state_dict(torch.load(ckpt_path), strict=True)
    net.eval()
    net.to(device)
    if os.path.exists(input_dir) and os.path.isdir(input_dir):
        for root, directory, filenames in os.walk(input_dir):
            print(f"正在解析{root}...")
            sub_dir = root[root.find(input_dir) + len(input_dir):]
            sub_dir = sub_dir.strip('/').strip(r'\\')
            samples = [s for s in glob(os.path.join(root, '*.jpg')) + glob(os.path.join(root, '*.png')) if
                       os.path.basename(s) != '__MASK__.png']
            total_samples = len(samples)
            if not overwrite:
                samples = [s for s in samples
                           if not os.path.exists(os.path.join(output_dir, sub_dir, os.path.basename(s)))]
            print(f"\toverwrite: {overwrite}, 一共{total_samples}个样本，变成{len(samples)}。")
            if samples:
                # patches_norm(samples, net, save_dir=os.path.join(output_dir, sub_dir), method=method)
                patches_norm_dataloader(samples, net, save_dir=os.path.join(output_dir, sub_dir), method=method,
                                        num_workers=num_workers, batch_size=batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('将所有的数据进行标准化！')
    parser.add_argument("--model_path", default=None, help="模型位置")
    parser.add_argument("--method", default='gray', choices=['rgb', 'gray'], help="标准化方法，可以为rgb或者gray。")
    parser.add_argument("--overwrite", default=False, action='store_true', help="是否使用缓存机制")
    args = parser.parse_args()
    i_dir = r'F:\20230802-WX-MrZ\patches'
    o_dir = r'F:\20230802-WX-MrZ\patches_norm'
    for i_dir, o_dir in [  # (r'F:\20230802-WX-MrZ\patches', r'F:\20230802-WX-MrZ\patches_norm'),
        (r'F:\20230629-PanWeiJun\patches', r'F:\20230629-PanWeiJun\patches_norm')]:
        main(i_dir, o_dir, model_path=r'C:\OnekeyPlatform\pretrain/patch_GAN.pth',
             method=['gray', 'rgb'].index(args.method), overwrite=args.overwrite,
             num_workers=1, batch_size=1)
    # input('按任意键退出...')
