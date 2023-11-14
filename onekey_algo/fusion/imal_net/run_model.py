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
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from torch.nn import CrossEntropyLoss

from onekey_algo.fusion.imal_net.utils import get_data_files, get_default_transform, get_dataloader
from onekey_algo.utils import create_dir_if_not_exists
from onekey_algo.utils.about_log import logger
from onekey_core.core import create_optimizer, create_lr_scheduler
from onekey_core.models.fusion.imalNet import imalNet

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def train_one_epoch(train_loader, model, optimizer, lr_scheduler, loss_function, device, epoch, iters_verbose=10):
    with torch.autograd.set_detect_anomaly(True):
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step_start = time.time()
            step += 1
            print(batch_data["label"])
            inputs, masks, labels = (batch_data["image"].to(device),
                                     batch_data["mask"].to(device),
                                     batch_data["label"].to(device))
            optimizer.zero_grad()
            seg, clf, rad, final = model(inputs)
            seg_loss_function, clf_loss_function = loss_function
            seg_loss = seg_loss_function(seg, masks)
            clf_loss = 0.01 * clf_loss_function(clf, labels) + \
                       0.01 * clf_loss_function(rad, labels) + clf_loss_function(final, labels)
            loss = seg_loss + clf_loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            epoch_loss += loss.item()
            if step % iters_verbose == 0:
                print(f"epoch: {epoch}, step:{step}, train_loss: {loss.item():.4f}, "
                      f"seg_train_loss: {seg_loss.item():.4f}, clf_train_loss: {clf_loss.item():.4f}, "
                      f"lr: {lr_scheduler.get_last_lr()[0]:.6f}, "
                      f"step time: {(time.time() - step_start):.4f}")
        lr_scheduler.step()
        epoch_loss /= step
        print(f"epoch {epoch} average loss: {epoch_loss:.4f}")
        return epoch_loss


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

    seg_loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    clf_loss_function = CrossEntropyLoss()
    optimizer = create_optimizer(args.optimizer, parameters=model.parameters(), lr=args.init_lr, weight_decay=1e-5)
    lr_scheduler = create_lr_scheduler('cosine', optimizer, T_max=args.epochs * len(train_ds) // args.batch_size)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    save_dir = create_dir_if_not_exists(args.save_dir, add_date=True)
    save_dir = create_dir_if_not_exists(os.path.join(save_dir, args.model_name))
    metric_values = []
    best_metric = 0
    best_metric_epoch = 0
    for epoch in range(args.epochs):
        train_one_epoch(train_loader, model, optimizer, lr_scheduler, (seg_loss_function, clf_loss_function), device,
                        epoch + 1, args.iters_verbose)
        if epoch % args.val_interval == 0:
            metric_value, clf_acc, rad_acc, final_acc = evaluate(val_loader, model, dice_metric, device)
            metric_values.append(metric_value.aggregate().item())
            metric_value.reset()
            print(f"DICE: {metric_values[-1]}, Clf ACC: {clf_acc}, Rad ACC: {rad_acc}, Final ACC: {final_acc}")
            if final_acc > best_metric:
                best_metric = final_acc
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(save_dir, f"{args.model_name}.pth"))
                print("saved new best metric model")
            print(f"current epoch: {epoch + 1} current mean dice: {metric_values[-1]:.4f}"
                  f"\nbest mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}")


DATA_ROOT = r'C:\Users\onekey\Project\OnekeyDS\CT\crop_3d'
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument('--train', nargs='*', default=DATA_ROOT, help='Training dataset')
    parser.add_argument('--valid', nargs='*', default=None, help='Validation dataset')
    parser.add_argument('--roi_size', nargs='*', default=[48, 48, 16], type=int, help='ROI size')
    parser.add_argument('--val_size', default=0.1, type=float, help='如果测试集为None，则训练集随机划分。')
    parser.add_argument('--num_classes', default=2, type=int, help='ROI中标签个数')
    parser.add_argument('-j', '--worker', dest='j', default=0, type=int, help='Number of workers.(default=1)')
    parser.add_argument('--model_name', default='imalNet', help='Model name')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to be used!')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--init_lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--optimizer', default='adam', help='Optimizer')
    parser.add_argument('--retrain', default=None, help='Retrain from path')
    parser.add_argument('--save_dir', default='.', help='path where to save')
    parser.add_argument('--iters_verbose', default=1, type=int, help='print frequency')
    parser.add_argument('--val_interval', default=1, type=int, help='print frequency')
    parser.add_argument('--cached_ratio', default=None, type=float, help='cached ratio')
    main(parser.parse_args())
