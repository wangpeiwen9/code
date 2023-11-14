# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2021/04/25
# Home: www.medai.icu
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2 019 All Rights Reserved.
r"""PyTorch Detection Training.

To run in a multi-gpu environment, use the distributed launcher::

    python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env \
        train.py ... --world-size $NGPU

The default hyperparameters are tuned for training on 8 gpus and 2 images per gpu.
    --lr 0.02 --batch-size 2 --world-size 8
If you use different number of gpus, the learning rate should be changed to 0.02/8*$NGPU.

On top of that, for training Faster/Mask R-CNN, the default hyperparameters are
    --epochs 26 --lr-steps 16 22 --aspect-ratio-group-factor 3

Also, if you train Keypoint R-CNN, the default hyperparameters are
    --epochs 46 --lr-steps 36 43 --aspect-ratio-group-factor 3
Because the number of images is smaller in the person keypoint subset of COCO,
the number of epochs should be adapted so that we have the same number of iterations.
# """
import argparse
import datetime
import json
import os
import shutil
import time

import torch
import torch.utils.data

from onekey_algo.datasets.PennFudanDataset import create_penn_dataset
from onekey_algo.detection import presets
from onekey_algo.detection.coco_eval import EVAL_METRIC_NAMES
from onekey_algo.detection.coco_utils import get_coco_kp, get_coco, get_coco_format, get_voc_format
from onekey_algo.detection.engine import train_one_epoch, evaluate
from onekey_algo.detection.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from onekey_core.core import create_optimizer, create_lr_scheduler, create_model
from onekey_algo.utils import create_dir_if_not_exists, truncate_dir
from onekey_algo.utils import detection_utils
from onekey_algo.utils.about_log import log_long_str, logger
from onekey_algo.utils.detection_utils import init_distributed_mode

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def get_dataset(data_path, name, image_set, transform):
    if name == 'penn':
        ds = create_penn_dataset(data_path, transform, task='detection')
        return ds, ds.num_classes
    elif name.startswith('coco'):
        paths = {
            "coco": (data_path, get_coco, 91),
            "coco_kp": (data_path, get_coco_kp, 2)
        }
        p, ds_fn, num_classes = paths[name]
        ds = ds_fn(p, image_set=image_set, transforms=transform)
    elif name.endswith('_coco_fmt'):
        with open(os.path.join(data_path, 'obj_labels.txt')) as f:
            labels = [l.strip() for l in f.readlines()]
        num_classes = len(labels)
        anno_exists = os.path.exists(os.path.join(data_path, image_set, 'annotations.json'))
        ds = get_coco_format(data_path, image_set=image_set, transforms=transform,
                             train_dir='' if anno_exists else None,
                             val_dir='' if anno_exists else None)
    elif name.endswith('_voc_fmt'):
        with open(os.path.join(data_path, 'obj_labels.txt')) as f:
            labels = [l.strip() for l in f.readlines()]
        num_classes = len(labels)
        ds = get_voc_format(data_path, image_set, transform, class_names=labels)
    else:
        raise ValueError(f'Dataset name {name} not found!')
    return ds, num_classes


def get_transform(train, data_augmentation):
    return presets.DetectionPresetTrain(data_augmentation) if train else presets.DetectionPresetEval()


def main(args):
    # print(vars(args))
    args.save_dir = create_dir_if_not_exists(args.save_dir, add_date=args.add_date)
    args.save_dir = os.path.join(args.save_dir, args.model)
    truncate_dir(args.save_dir, del_directly=True)
    train_dir = create_dir_if_not_exists(os.path.join(args.save_dir, 'train'))
    val_dir = create_dir_if_not_exists(os.path.join(args.save_dir, 'valid'))
    viz_dir = create_dir_if_not_exists(os.path.join(args.save_dir, 'viz'))
    init_distributed_mode(args)
    # print(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task = {'model_name': args.model, 'device': str(device), 'type': 'where'}

    # Data loading code
    dataset, num_classes = get_dataset(args.data_path,
                                       args.dataset, "train", get_transform(True, args.data_augmentation))
    task['num_classes'] = num_classes
    with open(os.path.join(viz_dir, 'task.json'), 'w') as task_file:
        print(json.dumps(task, ensure_ascii=False, indent=True), file=task_file)
    dataset_test, _ = get_dataset(args.data_path, args.dataset, "val", get_transform(False, args.data_augmentation))
    shutil.copy(os.path.join(args.data_path, 'obj_labels.txt'), os.path.join(viz_dir, 'labels.txt'))
    log_long_str(dataset)
    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    if args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=args.workers, collate_fn=detection_utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                                   sampler=test_sampler, num_workers=0,
                                                   collate_fn=detection_utils.collate_fn)

    print("Creating model")
    kwargs = {"trainable_backbone_layers": args.trainable_backbone_layers}
    if "rcnn" in args.model:
        if args.rpn_score_thresh is not None:
            kwargs["rpn_score_thresh"] = args.rpn_score_thresh

    model = create_model(f'detection.{args.model}', num_classes=num_classes, pretrained=args.pretrained, **kwargs)
    # model = torchvision.models.detection.__dict__[args.model](num_classes=num_classes, pretrained=args.pretrained,
    #                                                           **kwargs)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = create_optimizer('sgd', parameters=params, lr=args.lr,
                                 momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = create_lr_scheduler('mstep', optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Successfully load model from {args.resume}")

    if args.test_only:
        evaluate(model, data_loader_test, device=device)
        return

    print("Start training")
    start_time = time.time()
    eval_results_file = open(os.path.join(viz_dir, 'BST_RESULTS.txt'), 'w')
    print(','.join(['Epoch'] + EVAL_METRIC_NAMES), file=eval_results_file)
    training_log = open(os.path.join(viz_dir, 'training_log.txt'), 'w')
    print('iters,lr,loss,loss_classifier,loss_box_reg,loss_objectness,loss_rpn_box_reg', file=training_log)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq,
                        log_dir=training_log)
        lr_scheduler.step()
        if args.save_dir:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args': args,
                'epoch': epoch
            }
            detection_utils.save_on_master(checkpoint, os.path.join(train_dir, 'model_{}.pth'.format(epoch)))
            detection_utils.save_on_master(checkpoint, os.path.join(viz_dir, 'BEST-training-params.pth'))

        # evaluate after every epoch
        _, bbox_eval = evaluate(model, data_loader_test, epoch, device=device, log_dir=val_dir)
        print(bbox_eval, file=eval_results_file)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


# Modify this parameter if necessary!
DATA_ROOT = os.path.expanduser(r'~/Project/data/covid19')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Detection Training')

    parser.add_argument('--data-path', default=DATA_ROOT, help='Data root')
    parser.add_argument('--dataset', default='covid_coco_fmt', help='dataset')
    parser.add_argument('--model', default='fasterrcnn_resnet50_fpn', help='model')
    parser.add_argument('-b', '--batch-size', default=2, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--epochs', default=26, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--lr', default=0.02, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--add_date', default=False, action='store_true', help='是否在model_root下添加日期。')
    parser.add_argument('--lr-step-size', default=8, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-steps', default=[16, 22], nargs='+', type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--save_dir', default='.', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument('--rpn-score-thresh', default=None, type=float, help='rpn score threshold for faster-rcnn')
    parser.add_argument('--trainable-backbone-layers', default=None, type=int,
                        help='number of trainable layers of backbone')
    parser.add_argument('--data-augmentation', default="hflip", help='data augmentation policy (default: hflip)')
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
    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    main(parser.parse_args())
