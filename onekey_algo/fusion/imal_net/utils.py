# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2022/3/4
# Home: www.medai.icu
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2022 All Rights Reserved.
import glob
import os

from monai.data import CacheDataset, Dataset
from monai.data import DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    Orientationd,
    EnsureTyped,
    AddChanneld, Invertd, AsDiscreted, EnsureChannelFirstd, Spacingd, ScaleIntensityRanged, CropForegroundd, Resized,
)

from onekey_algo.custom.components.Radiology import diagnose_3d_image_mask_settings
from onekey_algo.utils.about_log import logger


def get_data_files(data_dir, val_size=0.2, force_valid: bool = True):
    train_images = sorted(glob.glob(os.path.join(data_dir, "images", "*.nii*")))
    train_labels = sorted(glob.glob(os.path.join(data_dir, "masks", "*.nii*")))
    with open(os.path.join(data_dir, 'label.txt')) as f:
        train_label_clf = {l.strip(',').split()[0]: l.strip().split()[1] for l in f.readlines()}
    assert len(train_images) != 0, f'{data_dir}没有找到任何数据，请检查数据。'
    assert len(train_images) == len(train_labels) == len(train_label_clf), "图像和Mask数量必须相同，请检查数据。"
    if force_valid:
        errors = diagnose_3d_image_mask_settings(train_images, train_labels, verbose=True)
        if errors:
            raise ValueError('数据检查出错，请核实数据格式。')
    data_dicts = [
        {"image": image_name, "mask": label_name, 'label': int(train_label_clf[os.path.basename(image_name)])}
        for image_name, label_name in zip(train_images, train_labels)
    ]
    val_num = int(len(train_images) * val_size)
    train_files, val_files = data_dicts[:-val_num], data_dicts[-val_num:]
    assert len(train_files) and len(val_files), f'训练集和测试集数量不能为0，请检查数据划分比例设置，或者数据量！'
    return train_files, val_files


def get_default_transform(roi_size=None):
    if roi_size is None:
        roi_size = [32, 32, 32]
    train_transform = Compose(
        [
            LoadImaged(keys=["image", "mask"]),
            AddChanneld(keys=["image", "mask"]),
            Orientationd(keys=["image", "mask"], axcodes="RAS"),
            Resized(keys=["image", "mask"], spatial_size=roi_size),
            EnsureTyped(keys=["image", "mask", 'label']),
        ]
    )
    val_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            # Spacingd(keys=["image", "label"], pixdim=(1., 1., 1.0), mode=("bilinear", "nearest")),
            # ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True, ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            # NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
    return train_transform, train_transform


def get_valid_transform(val_transform=None, num_classes=2):
    if val_transform is None:
        val_transform = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
                ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True, ),
                CropForegroundd(keys=["image"], source_key="image"),
                EnsureTyped(keys=["image", "label"]),
            ]
        )
    post_ori_transform = Compose([
        EnsureTyped(keys="pred"),
        Invertd(
            keys="pred",
            transform=val_transform,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        AsDiscreted(keys="pred", argmax=True, to_onehot=num_classes),
        AsDiscreted(keys="label", to_onehot=num_classes),
    ])
    return val_transform, post_ori_transform


def get_post_transform(test_transform=None, num_classes=2):
    if test_transform is None:
        test_transform = Compose(
            [
                LoadImaged(keys="image"),
                EnsureChannelFirstd(keys="image"),
                Orientationd(keys=["image"], axcodes="RAS"),
                Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
                ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True, ),
                CropForegroundd(keys=["image"], source_key="image"),
                EnsureTyped(keys="image"),
            ]
        )
    post_transform = Compose([
        EnsureTyped(keys="pred"),
        Invertd(
            keys="pred",
            transform=test_transform,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        AsDiscreted(keys="pred", argmax=True, to_onehot=num_classes),
        # SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir="./out", output_postfix="seg", resample=False),
    ])
    return test_transform, post_transform


def get_dataloader(train_files, val_files, train_transform, val_transform,
                   train_batch_size=2, train_workers=0, val_batch_size=1, val_workers=0,
                   cached_ratio: float = None, with_ds: bool = True):
    if isinstance(cached_ratio, (float, int)):
        logger.info(f'使用数据缓存，缓存比例：{cached_ratio}')
        train_ds = CacheDataset(data=train_files, transform=train_transform, cache_rate=cached_ratio)
        val_ds = CacheDataset(data=val_files, transform=val_transform, cache_rate=cached_ratio)
    else:
        train_ds = Dataset(data=train_files, transform=train_transform)
        val_ds = Dataset(data=val_files, transform=val_transform)
    # Form dataloader.
    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, num_workers=train_workers)
    val_loader = DataLoader(val_ds, batch_size=val_batch_size, num_workers=val_workers)
    if with_ds:
        return train_loader, val_loader, train_ds, val_ds
    else:
        return train_loader, val_loader
