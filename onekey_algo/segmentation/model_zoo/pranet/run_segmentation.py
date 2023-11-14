# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2021/6/22
# Home: www.medai.icu
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2021 All Rights Reserved.

import argparse
import os

import torch
import torch.nn.functional as F
from onekey_core.core import create_model, create_optimizer
from torch.autograd import Variable

from onekey_algo.utils import create_dir_if_not_exists
from onekey_algo.utils.about_log import logger
from onekey_algo.segmentation.model_zoo.pranet.utils.dataloader import get_loader
from onekey_algo.segmentation.model_zoo.pranet.utils.utils import clip_gradient, adjust_lr, AvgMeter


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def train(train_loader, model, optimizer, epoch, opt):
    total_step = len(train_loader)
    model.train()
    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25]
    loss_record2, loss_record3, loss_record4, loss_record5 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # ---- rescale ----
            trainsize = int(round(opt.train_size * rate / 32) * 32)
            if rate != 1:
                images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2 = model(images)
            # ---- loss function ----
            loss5 = structure_loss(lateral_map_5, gts)
            loss4 = structure_loss(lateral_map_4, gts)
            loss3 = structure_loss(lateral_map_3, gts)
            loss2 = structure_loss(lateral_map_2, gts)
            loss = loss2 + loss3 + loss4 + loss5  # TODO: try different weights for loss
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_record2.update(loss2.data, opt.batch_size)
                loss_record3.update(loss3.data, opt.batch_size)
                loss_record4.update(loss4.data, opt.batch_size)
                loss_record5.update(loss5.data, opt.batch_size)
        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            logger.info('Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                        '[lateral-2: {:.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f}, lateral-5: {:0.4f}]'.
                        format(epoch, opt.epochs, i, total_step,
                               loss_record2.show(), loss_record3.show(), loss_record4.show(), loss_record5.show()))
    save_path = os.path.join(opt.save_dir, 'ckpt')
    create_dir_if_not_exists(save_path)
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), os.path.join(save_path, 'PraNet-%d.pth' % epoch))
        logger.info('[Saving Snapshot:]', save_path + 'PraNet-%d.pth' % epoch)


def main(opt):
    # ---- build models ----
    # torch.cuda.set_device(1)  # set your gpu device

    model = create_model('segmentation.PraNet').cuda()

    # ---- flops and params ----
    # from utils.utils import CalParams
    # x = torch.randn(1, 3, 352, 352).cuda()
    # CalParams(lib, x)

    params = model.parameters()
    optimizer = create_optimizer('Adam', parameters=params, lr=opt.lr)

    image_root = '{}/train/image/'.format(opt.data_path)
    gt_root = '{}/train/mask/'.format(opt.data_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batch_size, trainsize=opt.train_size,
                              num_workers=opt.workers)

    logger.info("Start Training...")

    for epoch in range(1, opt.epochs):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch, opt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=12, help='training batch size')
    parser.add_argument('--train_size', type=int, default=352, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--data_path', type=str, default='./data/TrainDataset', help='path to train dataset')
    parser.add_argument('--save_dir', type=str, default='PraNet_Res2Net')
    args = parser.parse_args()
    main(args)
