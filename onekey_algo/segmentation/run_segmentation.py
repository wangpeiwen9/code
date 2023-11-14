# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2021/04/25
# Home: www.medai.icu
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2019 All Rights Reserved.
import argparse
import datetime
import json
import os
import shutil

from onekey_algo.utils.about_log import log_long_str, logger

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import time

import torch
import torch.utils.data
from torch import nn

from onekey_algo.datasets.ImageMaskDataset import create_image_mask_dataset
from onekey_algo.datasets.PennFudanDataset import create_penn_dataset
from onekey_algo.datasets.XRayDataset import create_xray_dataset
from onekey_core.core import create_optimizer, create_lr_scheduler, create_model
from onekey_algo.segmentation import presets
from onekey_algo.segmentation.coco_utils import get_coco, get_coco_format
from onekey_algo.utils import create_dir_if_not_exists, truncate_dir
from onekey_algo.utils.segmentation_utils import ConfusionMatrix, MetricLogger, SmoothedValue, collate_fn, \
    init_distributed_mode, \
    save_on_master


def get_dataset(data_path, name, image_set, transform):
    if name == 'penn':
        ds = create_penn_dataset(data_path, transform=transform, task='segmentation')
        return ds, ds.num_classes
    elif name == 'coco':
        num_classes = 21
        ds = get_coco(data_path, image_set=image_set, transforms=transform)
        return ds, num_classes
    elif name.endswith('_coco_fmt'):
        with open(os.path.join(data_path, 'seg_labels.txt')) as f:
            labels = f.readlines()
        num_classes = len(labels)
        ds = get_coco_format(data_path, image_set=image_set, transforms=transform)
        return ds, num_classes
    elif name.startswith('xray_'):
        ds = create_xray_dataset(name[5:], data_path=data_path, transform=transform, semantic_masks=True)
        return ds, ds.num_classes
    elif name.endswith('_binary_image_mask'):
        ds = create_image_mask_dataset(data_path, transforms=transform)
        return ds, 2
    elif name.endswith('_general_image_mask'):
        with open(os.path.join(data_path, 'seg_labels.txt')) as f:
            labels = f.readlines()
        num_classes = len(labels)
        ds = create_image_mask_dataset(os.path.join(data_path, image_set), transforms=transform,
                                       label_map=None, num_classes=num_classes)
        return ds, num_classes
    else:
        raise ValueError(f'Dataset name {name} not found!')


def get_transform(train, ratio=1, base_size=None, crop_size=None, resize=False):
    base_size = base_size or int(520 * ratio)
    crop_size = crop_size or int(480 * ratio)

    return presets.SegmentationPresetTrain(base_size, crop_size) \
        if train else presets.SegmentationPresetEval(base_size, resize=resize)


def criterion(inputs, target):
    if isinstance(inputs, dict):
        losses = {}
        for name, x in inputs.items():
            losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)

        if len(losses) == 1:
            return losses['out']

        return losses['out'] + 0.5 * losses['aux']
    else:
        return nn.functional.cross_entropy(inputs, target, ignore_index=255)


def evaluate(model, data_loader, device, num_classes, print_freq=16):
    model.eval()
    confmat = ConfusionMatrix(num_classes)
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'
    # number = 0
    # transform1 = transforms.ToPILImage(mode="L")
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image, target = image.to(device), target.to(device)
            # print(image.shape, target.shape)
            output = model(image)
            if isinstance(output, dict):
                output = output['out']
            # target_arr = np.uint8(target.cpu().numpy()).squeeze()
            # predict_arr = np.uint8(output.argmax(1).cpu().numpy()).squeeze()
            # target_arr[target_arr == 1] = 255
            # predict_arr[predict_arr == 1] = 255
            # transform1(target_arr).save(f'models/{number}_target.png')
            # transform1(predict_arr).save(f'models/{number}_predict.png')
            # number += 1
            confmat.update(target.flatten(), output.argmax(1).flatten())

        confmat.reduce_from_all_processes()

    return confmat


def train_one_epoch(model, criterion_fn, optimizer, data_loader, lr_scheduler, device, epoch, print_freq, log_dir):
    model.train()
    metric_logger = MetricLogger(delimiter="  ", log_dir=log_dir, epoch=epoch)
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        # print(image.shape, target.shape)
        output = model(image)
        loss = criterion_fn(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])


def main(args):
    save_dir = create_dir_if_not_exists(args.save_dir, add_date=False)
    save_dir = os.path.join(save_dir, args.model)
    truncate_dir(save_dir, del_directly=True)
    train_dir = create_dir_if_not_exists(os.path.join(save_dir, 'train'))
    val_dir = create_dir_if_not_exists(os.path.join(save_dir, 'valid'))
    viz_dir = create_dir_if_not_exists(os.path.join(save_dir, 'viz'))
    init_distributed_mode(args)
    # print(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task = {'model_name': args.model, 'device': str(device), 'type': 'which',
            'aux_loss': args.aux_loss or args.pretrained}

    dataset, num_classes = get_dataset(args.data_path, args.dataset, 'train',
                                       transform=get_transform(train=True,
                                                               ratio=args.downsample_ratio,
                                                               base_size=args.base_size,
                                                               crop_size=args.crop_size))
    log_long_str(dataset)
    dataset_test, _ = get_dataset(args.data_path, args.dataset, 'val',
                                  transform=get_transform(train=False,
                                                          ratio=args.downsample_ratio,
                                                          base_size=args.base_size,
                                                          crop_size=args.crop_size,
                                                          resize=True if 'unet' in args.model.lower() else False))
    log_long_str(dataset_test)
    dataset = {
        'train': dataset,
        'valid': dataset_test
    }
    shutil.copy(os.path.join(args.data_path, 'seg_labels.txt'), os.path.join(viz_dir, 'labels.txt'))
    task['num_classes'] = num_classes
    with open(os.path.join(viz_dir, 'task.json'), 'w') as task_file:
        print(json.dumps(task, ensure_ascii=False, indent=True), file=task_file)

    if args.distributed:
        sampler = {'train': torch.utils.data.distributed.DistributedSampler(dataset['train']),
                   'valid': torch.utils.data.distributed.DistributedSampler(dataset['valid'])}
    else:
        sampler = {'train': torch.utils.data.RandomSampler(dataset['train']),
                   'valid': torch.utils.data.SequentialSampler(dataset['valid'])}

    data_loader = {x: torch.utils.data.DataLoader(dataset[x],
                                                  batch_size=args.batch_size if x == 'train' else args.val_batch_size,
                                                  sampler=sampler[x], num_workers=args.workers,
                                                  drop_last=True,
                                                  collate_fn=collate_fn)
                   for x in ['train', 'valid']}

    model = create_model(f'segmentation.{args.model}', num_classes=num_classes, aux_loss=args.aux_loss,
                         pretrained=args.pretrained)
    model.to(device)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        logger.info(f'成功加载{args.resume}参数，开始继续训练。')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.test_only:
        confmat = evaluate(model, dataset['valid'], device=device, num_classes=num_classes, print_freq=args.print_freq)
        log_long_str(confmat)
        return

    if 'unet' in args.model.lower():
        params_to_optimize = [
            {"params": [p for p in model_without_ddp.parameters() if p.requires_grad]},
        ]
    else:
        params_to_optimize = [
            {"params": [p for p in model_without_ddp.backbone.parameters() if p.requires_grad]},
            {"params": [p for p in model_without_ddp.classifier.parameters() if p.requires_grad]},
        ]
        if args.aux_loss:
            params = [p for p in model_without_ddp.aux_classifier.parameters() if p.requires_grad]
            params_to_optimize.append({"params": params, "lr": args.lr * 10})

    # Initialize optimizer and learning rate for this run
    optimizer = create_optimizer(args.optimizer, parameters=params_to_optimize, lr=args.lr,
                                 momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = create_lr_scheduler('cosine', optimizer,
                                       T_max=args.epochs * len(dataset['train']) // args.batch_size)

    start_time = time.time()
    bst_acc = 0
    bst_epoch = None
    training_log = open(os.path.join(viz_dir, 'training_log.txt'), 'w')
    print("iters,lr,loss", file=training_log)
    for epoch in range(args.epochs):
        if args.distributed:
            sampler['train'].set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader['train'], lr_scheduler,
                        device, epoch, args.print_freq, training_log)
        confmat = evaluate(model, data_loader['valid'], device=device,
                           num_classes=num_classes, print_freq=args.print_freq)
        acc_global, acc, iu, dice = confmat.compute()
        with open(os.path.join(val_dir, f'Epoch-{epoch}.txt'), 'w') as valid_file:
            print(json.dumps({'global_acc': acc_global.item() * 100,
                              'acc_per_class': (acc * 100).tolist(),
                              'iou_per_class': (iu * 100).tolist(),
                              'mIoU': iu.mean().item() * 100,
                              'Dice': (dice * 100).tolist(),
                              'mDice': dice.mean().item() * 100},
                             ensure_ascii=False, indent=True),
                  file=valid_file)
        log_long_str(confmat)
        if args.save_per_epoch:
            save_on_master(
                {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                },
                os.path.join(train_dir, 'model_{}.pth'.format(epoch))
            )
        if dice.mean().item() > bst_acc:
            bst_acc = dice.mean().item()
            bst_epoch = epoch
            save_on_master(
                {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                },
                os.path.join(viz_dir, 'BEST-training-params.pth')
            )
            shutil.copy(os.path.join(val_dir, f'Epoch-{epoch}.txt'), os.path.join(viz_dir, 'BST_VAL_RESULTS.txt'))
        logger.info(f'Best mean DICE: {bst_acc}, @ {bst_epoch} epoch')
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


# Modify this parameter if necessary!
DATA_ROOT = os.path.expanduser(r'~/Project/data/covid19')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Segmentation Training')
    parser.add_argument('--dataset', default='xray_covid-19', help='dataset')
    parser.add_argument('--data_path', default=DATA_ROOT, help='Root data path.')
    parser.add_argument('--model', default='fcn_resnet50', help='model')
    parser.add_argument('--aux-loss', action='store_true', help='auxiliary loss')
    parser.add_argument('-b', '--batch-size', default=4, type=int)
    parser.add_argument('-vb', '--val_batch-size', default=4, type=int)
    parser.add_argument('--epochs', default=30, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--optimizer', default='sgd', help='Optimizer')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--save_dir', default='.', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--save_per_epoch', default=False, action='store_true', help='save checkpoint each epoch!')
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )
    # distributed training_a_classifier parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training_a_classifier')

    main(parser.parse_args())
