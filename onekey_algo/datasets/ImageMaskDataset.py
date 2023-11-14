# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2021/5/28
# Home: www.medai.icu
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2021 All Rights Reserved.
import os
from typing import Callable, Union

import numpy as np
from PIL import Image
from scipy import ndimage
# from torch.utils import data
from torch.utils.data import Dataset

from onekey_algo.utils import check_directory_exists
from onekey_algo.utils.about_log import logger
from onekey_core.core.image_loader import image_loader_3d

__all__ = ['ImageMaskDataset', 'ImageMaskDataset3D', 'create_image_mask_dataset']


def _intersection_images_and_annotation(images, masks):
    images = {os.path.splitext(i)[0]: i for i in images}
    masks = {os.path.splitext(i)[0]: i for i in masks}
    joint_keys = set(images.keys()) & set(masks.keys())
    return zip(*[(images[jk], masks[jk]) for jk in joint_keys])


class ImageMaskDataset2D(Dataset):
    def __init__(self, root, transforms=None, image_dir='image', mask_dir='mask',
                 label_map: Callable[[Union[Image.Image, np.ndarray]], Union[Image.Image, np.ndarray]] = None,
                 num_classes=2):
        image_root = os.path.join(root, image_dir)
        gt_root = os.path.join(root, mask_dir)
        check_directory_exists(root, image_root, gt_root, prefixes='Data root')
        images, masks = _intersection_images_and_annotation(os.listdir(image_root), os.listdir(gt_root))
        self.images = [os.path.join(image_root, image) for image in images]
        self.gts = [os.path.join(gt_root, mask) for mask in masks]
        self.label_map = label_map
        self.transforms = transforms
        self.root = root
        self.n_classes = num_classes

        self.filter_files()

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        if self.transforms is not None:
            image, gt = self.transforms(image, gt)
        return image, gt

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    @staticmethod
    def rgb_loader(path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        img = Image.open(path)
        # return img.convert('1')
        if self.label_map and callable(self.label_map):
            img = self.label_map(img)
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
        return img

    def __len__(self):
        return len(self.images)

    @property
    def num_classes(self):
        return self.n_classes

    def __repr__(self):
        head = "Dataset is " + self.__class__.__name__
        body = [f"Number classes: {self.num_classes}", "Number of samples: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body.append(f'Transform {self.transforms}')
        lines = [head] + [f"\t{line}" for line in body]
        return '\n'.join(lines)


def binary_label_map(img):
    if isinstance(img, Image.Image):
        img = np.array(img)
    half_max_v = np.max(img) / 2
    img[img < half_max_v] = 0
    img[img >= half_max_v] = 1
    return Image.fromarray(img)


class ImageMaskDataset3D(Dataset):
    def __init__(self, root_dir, img_list, sets, strict: bool = False):
        if isinstance(img_list, str) and os.path.exists(img_list):
            with open(img_list, 'r') as f:
                self.img_list = [line.strip().split() for line in f]
        elif isinstance(img_list, (list, tuple)):
            self.img_list = img_list
        else:
            raise ValueError(f"数据类型不支持，请检查训练数据设置！")
        self.root_dir = root_dir
        logger.info("读取到了 {} 数据".format(len(self.img_list)))
        # 获取检查通过的数据
        self.check_errors = []
        # self.get_valid_data(rm_errors=True)
        if strict and self.check_errors:
            for idx, error in enumerate(self.check_errors):
                logger.error(f"问题{idx + 1}：{error}")
            raise ValueError("检查失败，请结合问题和方法处理自己的数据！")
        self.input_D = sets.input_D
        self.input_H = sets.input_H
        self.input_W = sets.input_W
        self.phase = sets.phase

    def get_valid_data(self, rm_errors: bool = False):
        ilist = []
        logger.info('正在检查数据，请稍候...')
        for i, m in self.img_list:
            img = image_loader_3d(i, root=self.root_dir, index_order='C')
            msk = image_loader_3d(m, root=self.root_dir, index_order='C')
            if img is not None and msk is not None and img.shape == msk.shape and np.sum(msk) > 0:
                ilist.append([i, m])
            else:
                self.check_errors.append(f'{i}, {m} Check error!')
                os.remove(os.path.join(self.root_dir, i))
                os.remove(os.path.join(self.root_dir, m))
            logger.info(f'正在检查{i} 和 {m}对')
        logger.info(f'\t{len(ilist)}个样本通过检查。')
        self.img_list = ilist

    @staticmethod
    def __nii2tensorarray__(data):
        [z, y, x] = data.shape
        new_data = np.reshape(data, [1, z, y, x])
        new_data = new_data.astype("float32")

        return new_data

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if self.phase == "train":
            # read image and labels
            img_name, label_name = self.img_list[idx]
            img_name = os.path.join(self.root_dir, img_name)
            label_name = os.path.join(self.root_dir, label_name)
            assert os.path.isfile(img_name)
            assert os.path.isfile(label_name)
            img = image_loader_3d(img_name, index_order='C')  # We have transposed the data from WHD format to DHW
            assert img is not None
            mask = image_loader_3d(label_name, index_order='C')
            assert mask is not None

            # data processing
            img_array, mask_array = self.__training_data_process__(img, mask)

            # 2 tensor array
            img_array = self.__nii2tensorarray__(img_array)
            mask_array = self.__nii2tensorarray__(mask_array)

            assert img_array.shape == mask_array.shape, "img shape:{} is not equal to mask shape:{}".format(
                img_array.shape, mask_array.shape)
            logger.info(f'{label_name}全部ROI区域的体素大小为{np.sum(mask_array)}')
            return img_array, mask_array

        elif self.phase == "test":
            # read image
            img_name, _ = self.img_list[idx]
            img_name = os.path.join(self.root_dir, img_name)
            assert os.path.isfile(img_name)
            img = image_loader_3d(img_name, index_order='C')
            assert img is not None

            # data processing
            img_array = self.__testing_data_process__(img)

            # 2 tensor array
            img_array = self.__nii2tensorarray__(img_array)
            return img_array, img_name

    @staticmethod
    def __drop_invalid_range__(volume, label=None):
        """
        Cut off the invalid area
        """
        zero_value = volume[0, 0, 0]
        non_zeros_idx = np.where(volume != zero_value)

        [max_z, max_h, max_w] = np.max(np.array(non_zeros_idx), axis=1)
        [min_z, min_h, min_w] = np.min(np.array(non_zeros_idx), axis=1)

        if label is not None:
            return volume[min_z:max_z, min_h:max_h, min_w:max_w], label[min_z:max_z, min_h:max_h, min_w:max_w]
        else:
            return volume[min_z:max_z, min_h:max_h, min_w:max_w]

    @staticmethod
    def __random_center_crop__(data, label):
        from random import random
        """
        Random crop
        """
        target_indexes = np.where(label > 0)
        [img_d, img_h, img_w] = data.shape
        [max_D, max_H, max_W] = np.max(np.array(target_indexes), axis=1)
        [min_D, min_H, min_W] = np.min(np.array(target_indexes), axis=1)
        [target_depth, target_height, target_width] = np.array([max_D, max_H, max_W]) - np.array([min_D, min_H, min_W])
        Z_min = int((min_D - target_depth * 1.0 / 2) * random())
        Y_min = int((min_H - target_height * 1.0 / 2) * random())
        X_min = int((min_W - target_width * 1.0 / 2) * random())

        Z_max = int(img_d - ((img_d - (max_D + target_depth * 1.0 / 2)) * random()))
        Y_max = int(img_h - ((img_h - (max_H + target_height * 1.0 / 2)) * random()))
        X_max = int(img_w - ((img_w - (max_W + target_width * 1.0 / 2)) * random()))

        Z_min = np.max([0, Z_min])
        Y_min = np.max([0, Y_min])
        X_min = np.max([0, X_min])

        Z_max = np.min([img_d, Z_max])
        Y_max = np.min([img_h, Y_max])
        X_max = np.min([img_w, X_max])

        Z_min = int(Z_min)
        Y_min = int(Y_min)
        X_min = int(X_min)

        Z_max = int(Z_max)
        Y_max = int(Y_max)
        X_max = int(X_max)

        return data[Z_min: Z_max, Y_min: Y_max, X_min: X_max], label[Z_min: Z_max, Y_min: Y_max, X_min: X_max]

    @staticmethod
    def __intensity_normalize_one_volume__(volume):
        """
        normalize the intensity of an nd volume based on the mean and std of nonzero region
        inputs:
            volume: the input nd volume
        outputs:
            out: the normalized nd volume
        """

        pixels = volume[volume > 0]
        mean = pixels.mean()
        std = pixels.std()
        out = (volume - 140) / 200
        print(mean, std)
        out_random = np.random.normal(0, 1, size=volume.shape)
        out[volume == 0] = out_random[volume == 0]
        return out

    def __resize_data__(self, data):
        """
        Resize the data to the input size
        """
        [depth, height, width] = data.shape
        scale = [self.input_D * 1.0 / depth, self.input_H * 1.0 / height, self.input_W * 1.0 / width]
        data = ndimage.interpolation.zoom(data, scale, order=0)

        return data

    def __crop_data__(self, data, label):
        """
        Random crop with different methods:
        """
        # random center crop
        data, label = self.__random_center_crop__(data, label)

        return data, label

    def __training_data_process__(self, data, label):
        # drop out the invalid range
        data, label = self.__drop_invalid_range__(data, label)

        # crop data
        data, label = self.__crop_data__(data, label)

        # resize data
        data = self.__resize_data__(data)
        label = self.__resize_data__(label)

        # normalization data
        data = self.__intensity_normalize_one_volume__(data)

        return data, label

    def __testing_data_process__(self, data):
        # resize data
        data = self.__resize_data__(data)

        # normalization data
        data = self.__intensity_normalize_one_volume__(data)

        return data


def create_image_mask_dataset(root, transforms=None, image_dir='images', mask_dir='masks',
                              label_map=binary_label_map, **kwargs):
    return ImageMaskDataset2D(root, transforms=transforms, image_dir=image_dir, mask_dir=mask_dir,
                              label_map=label_map, **kwargs)
