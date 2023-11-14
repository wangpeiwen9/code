# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2022/3/4
# Home: www.medai.icu
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2022 All Rights Reserved.

import argparse
import os

import torch
from monai.metrics import DiceMetric

from onekey_algo.fusion.imal_net.utils import get_data_files, get_default_transform, get_dataloader
from onekey_algo.utils.about_log import logger
from onekey_core.models.fusion.imalNet import imalNet

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def evaluate(val_loader, model, metric, device):
    model.eval()
    clf_acc = 0
    rad_acc = 0
    final_acc = 0
    with torch.no_grad():
        for val_data in val_loader:
            inputs, masks, labels = (val_data["image"].to(device),
                                     val_data["mask"].to(device),
                                     val_data["label"].to(device))
            seg, clf, rad, final = model(inputs)
            clf_acc = clf_acc + torch.eq(torch.argmax(clf, dim=-1), labels)
            rad_acc = rad_acc + torch.eq(torch.argmax(rad, dim=-1), labels)
            final_acc = final_acc + torch.eq(torch.argmax(final, dim=-1), labels)
            metric(y_pred=seg, y=masks)
        return metric, float(clf_acc / len(val_loader)), float(rad_acc / len(val_loader)), \
               float(final_acc / len(val_loader))


def main(args, trans=None, ):
    assert args.cached_ratio is None or 0 <= args.cached_ratio <= 1, f"缓存比例必须介于 [0, 1]"
    train_trans, val_trans = get_default_transform(roi_size=(32, 32, 32))
    if args.valid is None:
        train_files, val_files = get_data_files(args.train, val_size=args.val_size, force_valid=False)
    else:
        train_files, _ = get_data_files(args.train, val_size=0)
        val_files, _ = get_data_files(args.valid, val_size=0)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f'一共获取{len(train_files)}个训练样本，{len(val_files)}个测试样本。')
    train_loader, val_loader, train_ds, val_ds = get_dataloader(train_files, val_files,
                                                                train_transform=train_trans, val_transform=val_trans,
                                                                train_batch_size=args.batch_size,
                                                                train_workers=args.j, val_workers=0,
                                                                cached_ratio=args.cached_ratio)

    model = imalNet(spatial_dims=3, in_channels=1, out_channels=64, strides=2, channels=[32, 64], num_classes=2)
    model = model.to(device)
    # Save the best training parameters.
    state_dict = torch.load(args.model_path, map_location=device)
    # load best core weights
    model.load_state_dict(state_dict)

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    metric_value, clf_acc, rad_acc, final_acc = evaluate(val_loader, model, dice_metric, device)
    print(f"DICE: {metric_value.aggregate().item()}, Clf ACC: {clf_acc}, Rad ACC: {rad_acc}, Final ACC: {final_acc}")


DATA_ROOT = r'C:\Users\onekey\Project\OnekeyDS\CT\crop_3d'
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument('--train', nargs='*', default=DATA_ROOT, help='Training dataset')
    parser.add_argument('--valid', nargs='*', default=None, help='Validation dataset')
    parser.add_argument('--model_path', default='./20221204/imalNet/imalNet.pth', help='ROI size')
    parser.add_argument('--val_size', default=0.1, type=float, help='如果测试集为None，则训练集随机划分。')
    parser.add_argument('--num_classes', default=2, type=int, help='ROI中标签个数')
    parser.add_argument('-j', '--worker', dest='j', default=0, type=int, help='Number of workers.(default=1)')
    parser.add_argument('--model_name', default='imalNet', help='Model name')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to be used!')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--cached_ratio', default=None, type=float, help='cached ratio')
    main(parser.parse_args())
