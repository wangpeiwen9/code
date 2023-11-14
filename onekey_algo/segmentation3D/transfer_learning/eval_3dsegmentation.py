# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2021/07/10
# Home: www.medai.icu
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2021 All Rights Reserved.
import os

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch.utils.data import DataLoader

from onekey_algo.datasets.ImageMaskDataset import ImageMaskDataset3D
from onekey_algo.segmentation3D.transfer_learning.model import generate_model
from onekey_algo.segmentation3D.transfer_learning.setting import parse_opts
from onekey_algo.utils.about_log import logger
from onekey_core.core.image_loader import image_loader_3d

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def seg_eval(pred, label, clss):
    """
    calculate the dice between prediction and ground truth
    input:
        pred: predicted mask
        label: groud truth
        clss: eg. [0, 1] for binary class
    """
    Ncls = len(clss)
    dices = np.zeros(Ncls)
    [depth, height, width] = pred.shape
    for idx, cls in enumerate(clss):
        # binary map
        pred_cls = np.zeros([depth, height, width])
        pred_cls[np.where(pred == cls)] = 1
        label_cls = np.zeros([depth, height, width])
        label_cls[np.where(label == cls)] = 1

        # cal the inter & conv
        s = pred_cls + label_cls
        inter = len(np.where(s >= 2)[0])
        conv = len(np.where(s >= 1)[0]) + inter
        try:
            dice = 2.0 * inter / conv
        except:
            print("conv is zeros when dice = 2.0 * inter / conv")
            dice = -1

        dices[idx] = dice

    return dices


def test(data_loader, model, img_names, sets):
    masks = []
    model.eval()  # for testing
    for batch_id, batch_data in enumerate(data_loader):
        # forward
        volume, img_name = batch_data
        logger.info(f'正在预测{img_name}...')
        if not sets.no_cuda:
            volume = volume.cuda()
        with torch.no_grad():
            probs = model(volume)
            probs = F.softmax(probs, dim=1)

        # resize mask to original size
        [batchsize, _, mask_d, mask_h, mask_w] = probs.shape
        data = image_loader_3d(os.path.join(sets.data_root, img_name[0]), index_order='C')
        [depth, height, width] = data.shape
        mask = probs[0]
        scale = [1, depth * 1.0 / mask_d, height * 1.0 / mask_h, width * 1.0 / mask_w]
        mask = ndimage.interpolation.zoom(mask.cpu(), scale, order=1)
        mask = np.argmax(mask, axis=0)
        masks.append(mask)

    return masks


def main(sets):
    # getting model
    checkpoint = torch.load(sets.resume_path)
    net, _ = generate_model(sets)
    net.load_state_dict(checkpoint['state_dict'])

    # data tensor
    testing_data = ImageMaskDataset3D(sets.data_root, sets.img_list, sets)
    data_loader = DataLoader(testing_data, batch_size=1, shuffle=False, num_workers=1, pin_memory=False)

    # testing
    if isinstance(sets.img_list, str) and os.path.exists(sets.img_list):
        with open(sets.img_list, 'r') as f:
            img_names, label_names = zip(*[line.strip() for line in f])
    elif isinstance(sets.img_list, (tuple, list)):
        img_names, label_names = zip(*sets.img_list)
    else:
        raise ValueError(f"数据类型不支持，请检查训练数据设置！")
    masks = test(data_loader, net, img_names, sets)
    for iname, mask_arr, lname in zip(img_names, masks, label_names):
        new_image = nib.Nifti1Image(np.transpose(mask_arr.astype(np.int16), [2, 1, 0]), np.eye(4))
        fname, _ = os.path.splitext(lname)
        nib.save(new_image, f'{fname}.nii.gz')
        print(f'{np.sum(mask_arr.astype(np.int16))}Saving {iname}')

    # evaluation: calculate dice
    Nimg = len(data_loader)
    dices = np.zeros([Nimg, sets.n_seg_classes])
    for idx in range(Nimg):
        label = image_loader_3d(os.path.join(sets.data_root, label_names[idx]), index_order='C')
        dices[idx, :] = seg_eval(masks[idx], label, range(sets.n_seg_classes))

    # print result
    for idx in range(0, sets.n_seg_classes):
        mean_dice_per_task = np.mean(dices[:, idx])
        print('mean dice for class-{} is {}'.format(idx, mean_dice_per_task))


if __name__ == '__main__':
    # settting
    sets = parse_opts()
    sets.phase = 'test'
    main(sets=sets)
