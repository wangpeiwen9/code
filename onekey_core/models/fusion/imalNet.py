# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2022/05/31
# Forum: www.medai.icu
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2022 All Rights Reserved.

from functools import partial
from typing import Callable, Union
from typing import Sequence

import SimpleITK as sitk
import numpy as np
import torch
import torch.nn as nn
from monai.losses import DiceCELoss
from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers import get_pool_layer
from monai.networks.layers.factories import Conv, Norm

__all__ = ["imalNet"]

from radiomics import featureextractor
from torch.nn import CrossEntropyLoss


class ResNetBlock(nn.Module):
    def __init__(
            self,
            in_planes: int,
            mid_planes: int,
            out_planes: int,
            spatial_dims: int = 3,
            stride: int = 1,
            downsample: Union[nn.Module, partial, None] = None,
            with_atten: bool = False
    ) -> None:
        """
        Args:
            in_planes: number of input channels.
            mid_planes: number of middle channels.
            out_planes: number of output channels.

            spatial_dims: number of spatial dimensions of the input image.
            stride: stride to use for first conv layer.
            downsample: which downsample layer to use.
        """
        super().__init__()

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        norm_type: Callable = Norm[Norm.BATCH, spatial_dims]

        self.conv1 = conv_type(in_planes, mid_planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = norm_type(mid_planes)
        self.conv2 = conv_type(mid_planes, mid_planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = norm_type(mid_planes)
        self.conv3 = conv_type(mid_planes, out_planes, kernel_size=3, padding=1, bias=False)
        self.bn3 = norm_type(out_planes)
        self.relu = nn.ReLU(inplace=False)

        # downsample = conv_type(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
        self.downsample = downsample
        self.stride = stride
        self.with_atten = with_atten

    def forward(self, x: torch.Tensor, atten: Union[torch.Tensor, None] = None) -> (torch.Tensor, torch.Tensor):
        residual = x

        out: torch.Tensor = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        l3 = self.bn3(out)

        atten_out = None
        if self.with_atten:
            atten_out = torch.softmax(l3, dim=1)
        if self.downsample is not None:
            residual = self.downsample(x)

        # 激活output
        out = out + residual
        if atten is not None:
            out = out + atten * l3
        out = self.relu(out)

        return out, atten_out


def get_up_layer(in_channels, out_channels, channels, strides, kernel_size, dropout=0):
    trans_conv1 = Convolution(3, in_channels, channels[-1],
                              strides=strides, kernel_size=kernel_size, dropout=dropout,
                              bias=False, conv_only=True, is_transposed=True)
    trans_conv2 = Convolution(3, channels[-1], channels[-2],
                              strides=strides, kernel_size=kernel_size, dropout=dropout,
                              bias=False, conv_only=True, is_transposed=True)
    trans_conv3 = Convolution(3, channels[-2], out_channels,
                              strides=strides, kernel_size=kernel_size, dropout=dropout,
                              bias=False, conv_only=True, is_transposed=True)
    return nn.Sequential(trans_conv1, trans_conv2, trans_conv3)


def get_radiomics_feature(img_: torch.Tensor, msk_: torch.Tensor, extractor, num_features) -> torch.Tensor:
    features = []
    img_ = torch.squeeze(img_, dim=1)
    msk_ = torch.squeeze(msk_, dim=1)
    for img, msk in zip(img_, msk_):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
            img = sitk.GetImageFromArray(img)
        if isinstance(msk, torch.Tensor):
            msk = msk.detach().cpu().numpy()
            msk = sitk.GetImageFromArray(msk)
        try:
            featureVector = extractor.execute(img, msk, label=1)
            features.append([float(featureVector[featureName]) if not np.isnan(featureVector[featureName]) else 0
                             for featureName in featureVector.keys() if 'diagnostics' not in featureName])
        except:
            print(f"使用全0填充...")
            features.append([0] * num_features)
    features = torch.from_numpy(np.array(features)).float().to('cuda:0')
    return features


class imalNet(nn.Module):
    def __init__(
            self,
            num_classes: int,
            num_rad_features: int = 107,
            kernel_size: Union[Sequence[int], int] = 3,
            **kwargs) -> None:
        super().__init__()

        conv_type: Callable = Conv[Conv.CONV, 3]
        self.seg_ds1 = conv_type(1, 64, kernel_size=1, stride=1, bias=False)
        self.seg1 = ResNetBlock(in_planes=1, mid_planes=32, out_planes=64, spatial_dims=3, stride=1,
                                downsample=self.seg_ds1, with_atten=True)
        self.clf_ds1 = conv_type(1, 64, kernel_size=1, stride=1, bias=False)
        self.clf1 = ResNetBlock(in_planes=1, mid_planes=32, out_planes=64, spatial_dims=3, stride=1,
                                downsample=self.clf_ds1)
        self.seg1_transition = conv_type(64, 64, kernel_size=3, stride=2, bias=False, padding=1)
        self.clf1_transition = conv_type(64, 64, kernel_size=3, stride=2, bias=False, padding=1)

        self.seg_ds2 = conv_type(64, 64, kernel_size=1, stride=1, bias=False)
        self.seg2 = ResNetBlock(in_planes=64, mid_planes=32, out_planes=64, spatial_dims=3, stride=1,
                                downsample=self.seg_ds2, with_atten=True)
        self.clf_ds2 = conv_type(64, 64, kernel_size=1, stride=1, bias=False)
        self.clf2 = ResNetBlock(in_planes=64, mid_planes=32, out_planes=64, spatial_dims=3, stride=1,
                                downsample=self.clf_ds2)
        self.seg2_transition = conv_type(64, 64, kernel_size=3, stride=2, bias=False, padding=1)
        self.clf2_transition = conv_type(64, 64, kernel_size=3, stride=2, bias=False, padding=1)

        self.seg_ds3 = conv_type(64, 64, kernel_size=1, stride=1, bias=False)
        self.seg3 = ResNetBlock(in_planes=64, mid_planes=32, out_planes=64, spatial_dims=3, stride=1,
                                downsample=self.seg_ds3, with_atten=True)
        self.clf_ds3 = conv_type(64, 64, kernel_size=1, stride=1, bias=False)
        self.clf3 = ResNetBlock(in_planes=64, mid_planes=32, out_planes=64, spatial_dims=3, stride=1,
                                downsample=self.clf_ds3)
        self.seg3_transition = conv_type(64, 64, kernel_size=3, stride=2, bias=False, padding=1)
        self.clf3_transition = conv_type(64, 64, kernel_size=3, stride=2, bias=False, padding=1)

        # 分割模型 Seg decoder
        self.seg_decoder = get_up_layer(64, 2, channels=[64, 32], strides=2, kernel_size=kernel_size)
        # 分类模型 Clf decoder
        self.clf_decoder = get_pool_layer(("adaptiveavg", {"output_size": (1, 1, 1)}), spatial_dims=3)
        self.clf_task = torch.nn.Linear(64, num_classes, bias=False)
        # 初始化传统组学特征提取
        self.num_rad_features = num_rad_features
        self.extractor = featureextractor.RadiomicsFeatureExtractor()

        # FSM模块
        self.fsm_act = torch.nn.SELU(inplace=True)
        self.fsm_bn = torch.nn.BatchNorm1d(num_rad_features)
        self.fsm1 = torch.nn.Linear(num_rad_features, 64, bias=False)
        self.fsm2 = torch.nn.Linear(64, num_rad_features, bias=False)
        # Rad任务模块
        self.rad_task = torch.nn.Linear(num_rad_features, num_classes, bias=False)

        # Final任务模块
        self.final_act = torch.nn.ReLU(inplace=True)
        num_concat_features = num_rad_features + 64
        self.fc1 = torch.nn.Linear(num_concat_features, 64, bias=False)
        self.fc2 = torch.nn.Linear(64, 32, bias=False)
        self.final_task = torch.nn.Linear(32, num_classes, bias=False)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        seg_out, seg_atten = self.seg1(x)
        seg_out = self.seg1_transition(seg_out)
        clf_out, _ = self.clf1(x, seg_atten)
        clf_out = self.clf1_transition(clf_out)

        seg_out, seg_atten = self.seg2(seg_out)
        seg_out = self.seg2_transition(seg_out)
        clf_out, _ = self.clf2(clf_out, seg_atten)
        clf_out = self.clf2_transition(clf_out)

        seg_out, seg_atten = self.seg3(seg_out)
        seg_out = self.seg3_transition(seg_out)
        clf_out, _ = self.clf3(clf_out, seg_atten)
        clf_out = self.clf3_transition(clf_out)

        mask = self.seg_decoder(seg_out)
        seg_task = torch.argmax(mask, 1, keepdim=True)
        clf_out = self.clf_decoder(clf_out)

        # FSM
        rad_features = get_radiomics_feature(x, seg_task, self.extractor, self.num_rad_features)
        rad_features = self.fsm_bn(rad_features)
        fsm = self.fsm_act(self.fsm1(rad_features))
        fsm = self.fsm_act(self.fsm2(fsm))
        fsm_out = torch.softmax(fsm, dim=1) * rad_features
        rad_task = self.rad_task(fsm_out)

        # Classification
        clf_out = torch.reshape(clf_out, (-1, 64))
        clf_task = self.clf_task(clf_out)

        # Final task
        concat_features = torch.cat([clf_out, fsm_out], dim=1)
        fc = self.final_act(self.fc1(concat_features))
        fc = self.final_act(self.fc2(fc))
        final_task = self.final_task(fc)
        return mask, clf_task, rad_task, final_task


if __name__ == '__main__':
    with torch.autograd.set_detect_anomaly(True):
        a = torch.rand((4, 1, 32, 32, 32))
        m = torch.randint(0, 2, (4, 1, 32, 32, 32))
        c = torch.randint(0, 2, (4,))
        imal_net = imalNet(spatial_dims=3, in_channels=1, out_channels=64, strides=2, channels=[32, 64], num_classes=2)
        seg, clf, rad, final = imal_net(a)
        print(a.shape, m.shape, seg.shape, clf.shape, rad.shape, final.shape, c)
        seg_loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
        clf_loss_function = CrossEntropyLoss()
        seg_loss = seg_loss_function(seg, m)
        clf_loss = clf_loss_function(clf, c) + clf_loss_function(rad, c) + clf_loss_function(final, c)
        loss = seg_loss + clf_loss
        print(seg_loss + clf_loss, seg_loss, clf_loss)
        loss.backward()
