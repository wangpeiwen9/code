# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2021/07/10
# Home: www.medai.icu
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2021 All Rights Reserved.

import os
import time

import numpy as np
import torch
from onekey_algo.datasets.ImageMaskDataset import ImageMaskDataset3D
from onekey_algo.segmentation3D.transfer_learning.model import generate_model
from onekey_algo.segmentation3D.transfer_learning.setting import parse_opts
from onekey_algo.utils.about_log import logger
from onekey_core.core.losses3D import DiceLoss, BCEDiceLoss
from scipy import ndimage
from torch import optim
from torch.utils.data import DataLoader

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def train(data_loader, model, optimizer, scheduler, total_epochs, save_interval, save_folder, sets):
    # settings
    batches_per_epoch = len(data_loader)
    logger.info('{} epochs in total, {} batches per epoch'.format(total_epochs, batches_per_epoch))
    # loss_seg = nn.CrossEntropyLoss(ignore_index=-1)
    loss_seg = BCEDiceLoss(classes=sets.n_seg_classes, alpha=0.1, weight=torch.tensor([[0.1], [1]]))
    # loss_seg = DiceLoss(classes=sets.n_seg_classes)
    logger.info(f"Current setting is:{sets}")
    if not sets.no_cuda:
        loss_seg = loss_seg.cuda()

    model.train()
    train_time_sp = time.time()
    total_batch_id = 0
    for epoch in range(total_epochs):
        logger.info('Start epoch {}'.format(epoch))
        logger.info('lr = {}'.format(scheduler.get_last_lr()))
        for batch_id, batch_data in enumerate(data_loader):
            total_batch_id += 1
            volumes, label_masks = batch_data

            if not sets.no_cuda:
                volumes = volumes.cuda()

            optimizer.zero_grad()
            out_masks = model(volumes)
            # resize label
            [n, _, d, h, w] = out_masks.shape
            new_label_masks = np.zeros([n, d, h, w])
            for label_id in range(n):
                label_mask = label_masks[label_id]
                [ori_c, ori_d, ori_h, ori_w] = label_mask.shape
                label_mask = np.reshape(label_mask, [ori_d, ori_h, ori_w])
                scale = [d * 1.0 / ori_d, h * 1.0 / ori_h, w * 1.0 / ori_w]
                label_mask = ndimage.interpolation.zoom(label_mask, scale, order=0)
                new_label_masks[label_id] = label_mask

            new_label_masks = torch.tensor(new_label_masks).to(torch.int64)
            if not sets.no_cuda:
                new_label_masks = new_label_masks.cuda()

            # calculating loss
            loss1, loss2, per_score = loss_seg(out_masks, new_label_masks)
            loss = loss1 + loss2
            # if isinstance(loss, tuple):
            #     loss = loss[0]
            loss.backward()
            optimizer.step()
            scheduler.step()

            avg_batch_time = (time.time() - train_time_sp) / total_batch_id
            logger.info(f'Batch: {epoch}-{batch_id} ({total_batch_id}), '
                        f'loss = {loss.item():.3f}, loss1 = {loss1.item():.3f}, loss2 = {loss2.item():.3f},'
                        f'avg_batch_time = {avg_batch_time:.3f}, score = {per_score}')

        if not sets.ci_test:
            # save model
            model_save_path = '{}_epoch_{}_batch_{}.pth'.format(save_folder, epoch, total_batch_id)
            model_save_dir = os.path.dirname(model_save_path)
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)

            logger.info('Save checkpoints: epoch = {}, batch_id = {}'.format(epoch, total_batch_id))
            torch.save({
                'epoch': epoch,
                'batch_id': total_batch_id,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()},
                model_save_path)

    print('Finished training')
    if sets.ci_test:
        exit()


def main(sets):
    if sets.ci_test:
        sets.img_list = './toy_data/test_ci.txt'
        sets.n_epochs = 1
        sets.no_cuda = True
        sets.data_root = './toy_data'
        sets.pretrain_path = ''
        sets.num_workers = 0
        sets.model_depth = 10
        sets.resnet_shortcut = 'A'
        sets.input_D = 14
        sets.input_H = 28
        sets.input_W = 28

    # getting model
    torch.manual_seed(sets.manual_seed)
    model, parameters = generate_model(sets)
    # print (model)
    # optimizer
    if sets.ci_test or not sets.pretrain_path:
        params = [{'params': parameters, 'lr': sets.learning_rate}]
    else:
        params = [{'params': parameters['base_parameters'], 'lr': sets.learning_rate},
                  {'params': parameters['new_parameters'], 'lr': sets.learning_rate * 100}]
    optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # train from resume
    if sets.resume_path:
        if os.path.isfile(sets.resume_path):
            logger.info("=> loading checkpoint '{}'".format(sets.resume_path))
            checkpoint = torch.load(sets.resume_path)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(sets.resume_path, checkpoint['epoch']))

    # getting data
    sets.phase = 'train'
    if sets.no_cuda:
        sets.pin_memory = False
    else:
        sets.pin_memory = True
    training_dataset = ImageMaskDataset3D(sets.data_root, sets.img_list, sets)
    data_loader = DataLoader(training_dataset, batch_size=sets.batch_size, shuffle=True, num_workers=sets.num_workers,
                             pin_memory=sets.pin_memory)

    # training
    train(data_loader, model, optimizer, scheduler, total_epochs=sets.n_epochs, save_interval=sets.save_intervals,
          save_folder=sets.save_folder, sets=sets)


if __name__ == '__main__':
    # settting
    sets = parse_opts()
    main(sets)
