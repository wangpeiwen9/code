# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2022/4/14
# Home: www.medai.icu
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2022 All Rights Reserved.
import glob
import os

import nibabel as nib
import numpy as np
import torch
from monai.data import DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference

from onekey_algo import OnekeyDS
from onekey_algo.segmentation3D.modelzoo import DEFAULT_MODEL_PARAMS
from onekey_algo.segmentation3D.modelzoo.utils import get_post_transform
from onekey_algo.utils import create_dir_if_not_exists
from onekey_algo.utils.about_log import logger
from onekey_core.core import create_model

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def fix_spacing(o, r, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    try:
        img = nib.load(o)
        w = nib.load(r)
        qform = img.get_qform()
        w.set_qform(qform)
        sform = img.get_sform()
        w.set_sform(sform)
        nib.save(w, os.path.join(save_dir, os.path.basename(r)))
        # # print(f"正在转化{o}，{r}...")
        # origin_img = sitk.ReadImage(o)  # 读取原始文件
        # origin = origin_img.GetOrigin()  # 这三句是获取的原始图像文件的位置和方向吧。
        # spacing = origin_img.GetSpacing()
        # direction = origin_img.GetDirection()
        #
        # w = sitk.ReadImage(r)  # 读取自己预测得到的nii.gz文件
        #
        # w.SetOrigin(origin)  # 将自己的文件处理成和官方一致的位置坐标系
        # w.SetSpacing(spacing)
        # w.SetDirection(direction)
        # sitk.WriteImage(w, os.path.join(save_dir, os.path.basename(r)))  # 处理完之后保存到相应的合适位置。
    except Exception as e:
        print(f'\t转化{o}，{r}遇到错误{e}')


def inference(data, model, transformer, device, roi_size=(48, 48, 48), save_dir='.', save_name=None):
    val_trans, ori_trans = transformer
    create_dir_if_not_exists(save_dir)
    if not isinstance(data, (list, tuple)):
        data = [data]
    assert all(os.path.exists(d) for d in data)
    data = [{'image': d} for d in data]
    val_org_ds = Dataset(data=data, transform=val_trans)
    val_org_loader = DataLoader(val_org_ds, batch_size=1, num_workers=0)
    with torch.no_grad():
        for val_data, fname in zip(val_org_loader, data):
            logger.info(f'正在预测{fname}')
            val_inputs = val_data["image"].to(device)
            sw_batch_size = 1
            val_data["pred"] = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)

            val_data = [ori_trans(i) for i in decollate_batch(val_data)]
            # print(np.array(val_data[0]['pred']).shape)
            img = nib.Nifti1Image(np.squeeze(np.array(val_data[0]['pred'])).astype(np.int16), np.eye(4))
            if save_name is None:
                save_name_ = os.path.join(save_dir, os.path.basename(fname['image']))
            else:
                save_name_ = os.path.join(save_dir, f"{save_name}")
            nib.save(img, save_name_)
            fix_spacing(fname['image'], save_name_, save_dir=save_dir)


def init(model_name, model_path, num_classes=2, roi_size=(48, 48, 48), device='cpu', **kwargs):
    model_params = DEFAULT_MODEL_PARAMS[model_name.upper()]
    model_params.update({'out_channels': num_classes, 'roi_size': roi_size})
    model_params.update(kwargs)
    model = create_model(f'segmentation3d.{model_name}', **model_params)
    device = device or torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(model_path, map_location=device)
    if 'state_dict' in state_dict:
        model.load_state_dict(state_dict['state_dict'])
    else:
        model.load_state_dict(state_dict)
    model = model.to(device)
    val_trans, ori_trans = get_post_transform()
    logger.info(f'使用{model_name}模型，成功加载{model_path}参数')
    return model, (val_trans, ori_trans), device


if __name__ == "__main__":
    data = [i for i in glob.glob(os.path.join(OnekeyDS.ct, 'images', '*.nii*'))]
    m, t, d = init('Unet', model_path='20220414/Unet/Unet.pth', num_classes=2)
    inference(data, m, t, d, roi_size=(48, 48, 48), save_dir='.')
