# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2022/3/4
# Home: www.medai.icu
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2022 All Rights Reserved.

import argparse
import os
import time

import torch
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete

from onekey_algo import OnekeyDS
from onekey_algo.segmentation3D.modelzoo import DEFAULT_MODEL_PARAMS
from onekey_algo.segmentation3D.modelzoo.utils import get_data_files, get_default_transform, get_dataloader
from onekey_algo.utils import create_dir_if_not_exists
from onekey_algo.utils.about_log import logger
from onekey_core.core import create_model, create_optimizer, create_lr_scheduler

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def train_one_epoch(train_loader, model, optimizer, lr_scheduler, loss_function, device, epoch, iters_verbose=10):
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step_start = time.time()
        step += 1
        inputs, labels = (batch_data["image"].to(device), batch_data["label"].to(device))
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        epoch_loss += loss.item()
        if step % iters_verbose == 0:
            print(f"epoch: {epoch}, step:{step}, train_loss: {loss.item():.4f}, "
                  f"lr: {lr_scheduler.get_last_lr()[0]:.6f}, "
                  f"step time: {(time.time() - step_start):.4f}")
    lr_scheduler.step()
    epoch_loss /= step
    print(f"epoch {epoch} average loss: {epoch_loss:.4f}")
    return epoch_loss


def evaluate(val_loader, model, post_trans, metric, device, roi_size=(48, 48, 48)):
    model.eval()
    post_pred, post_label = post_trans
    with torch.no_grad():
        for val_data in val_loader:
            val_inputs, val_labels = (val_data["image"].to(device), val_data["label"].to(device),)
            val_outputs = sliding_window_inference(inputs=val_inputs, roi_size=roi_size, sw_batch_size=1,
                                                   predictor=model, overlap=0.5, )
            val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
            val_labels = [post_label(i) for i in decollate_batch(val_labels)]
            # print(val_outputs[0].shape, val_labels[0].shape)
            metric(y_pred=val_outputs, y=val_labels)

        return metric


def main(args, trans=None, ):
    if len(args.roi_size) == 3:
        roi_size = args.roi_size
    elif len(args.roi_size) == 1:
        roi_size = args.roi_size * 3
    else:
        raise ValueError('`roi_size`必须长度为1，或者3.')
    assert args.cached_ratio is None or 0 <= args.cached_ratio <= 1, f"缓存比例必须介于 [0, 1]"
    if trans is None:
        logger.info('使用自定义Transform')
        train_trans, val_trans = get_default_transform(roi_size=roi_size)
    else:
        logger.info('使用Default Transform')
        train_trans, val_trans = trans
    if args.valid is None:
        train_files, val_files = get_data_files(args.train, val_size=args.val_size, force_valid=False)
    else:
        train_files, _ = get_data_files(args.train, val_size=0, force_valid=False)
        val_files, _ = get_data_files(args.valid, val_size=0, force_valid=False)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    # 后处理
    post_label = AsDiscrete(to_onehot=args.num_classes)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.num_classes)
    # print(json.dumps(val_files, ensure_ascii=False, indent=True))
    logger.info(f'一共获取{len(train_files)}个训练样本，{len(val_files)}个测试样本。')
    train_loader, val_loader, train_ds, val_ds = get_dataloader(train_files, val_files,
                                                                train_transform=train_trans, val_transform=val_trans,
                                                                train_batch_size=args.batch_size,
                                                                train_workers=args.j, val_workers=0,
                                                                cached_ratio=args.cached_ratio)
    model_params = DEFAULT_MODEL_PARAMS[args.model_name.upper()]
    model_params.update({'out_channels': args.num_classes, 'img_size': roi_size})
    model = create_model(f'segmentation3d.{args.model_name}', **model_params)
    model = model.to(device)

    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = create_optimizer(args.optimizer, parameters=model.parameters(), lr=args.init_lr, weight_decay=1e-5)
    lr_scheduler = create_lr_scheduler('cosine', optimizer, T_max=args.epochs * len(train_ds) // args.batch_size)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    save_dir = create_dir_if_not_exists(args.save_dir, add_date=True)
    save_dir = create_dir_if_not_exists(os.path.join(save_dir, args.model_name))
    metric_values = []
    best_metric = 0
    best_metric_epoch = 0
    for epoch in range(args.epochs):
        train_one_epoch(train_loader, model, optimizer, lr_scheduler, loss_function, device, epoch + 1,
                        args.iters_verbose)
        if epoch % args.val_interval == 0:
            metric_value = evaluate(val_loader, model, (post_pred, post_label), dice_metric, device, roi_size=roi_size)
            metric_values.append(metric_value.aggregate().item())
            metric_value.reset()

            if metric_values[-1] > best_metric:
                best_metric = metric_values[-1]
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(save_dir, f"{args.model_name}.pth"))
                print("saved new best metric model")
            print(f"current epoch: {epoch + 1} current mean dice: {metric_values[-1]:.4f}"
                  f"\nbest mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}")


DATA_ROOT = OnekeyDS.ct
# DATA_ROOT = '/Users/zhangzhiwei/Downloads/Abdomen/RawData/Training'
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument('--train', nargs='*', default=DATA_ROOT, help='Training dataset')
    parser.add_argument('--valid', nargs='*', default=None, help='Validation dataset')
    parser.add_argument('--roi_size', nargs='*', default=[48, 48, 16], type=int, help='ROI size')
    parser.add_argument('--val_size', default=0.1, type=float, help='如果测试集为None，则训练集随机划分。')
    parser.add_argument('--num_classes', default=2, type=int, help='ROI中标签个数')
    parser.add_argument('-j', '--worker', dest='j', default=0, type=int, help='Number of workers.(default=1)')
    parser.add_argument('--model_name', default='Unet', help='Model name')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to be used!')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--init_lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--optimizer', default='adam', help='Optimizer')
    parser.add_argument('--retrain', default=None, help='Retrain from path')
    parser.add_argument('--save_dir', default='.', help='path where to save')
    parser.add_argument('--iters_verbose', default=10, type=int, help='print frequency')
    parser.add_argument('--val_interval', default=4, type=int, help='print frequency')
    parser.add_argument('--cached_ratio', default=None, type=float, help='cached ratio')
    main(parser.parse_args())
