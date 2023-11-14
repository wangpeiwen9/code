# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2022/3/22
# Home: www.medai.icu
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2022 All Rights Reserved.
from typing import Union

import cv2
import nibabel as nib
import numpy as np
from skimage.feature import canny

FILL_MODE_CHUNK = 'CHUNK'
FILL_MODE_LINE = 'LINE'


def _get_coordinates(h, w, h_max, w_max, dir):
    coordinates_h = np.clip([h - dir, h, h + dir, h + dir, h + dir], 0, h_max - 1)
    coordinates_w = np.clip([w + dir, w + dir, w - dir, w, w + dir], 0, w_max - 1)
    return zip(coordinates_h, coordinates_w)


def pad(data: np.ndarray, when: Union[float, int] = 1, value: Union[float, int, None] = None, type: str = 'out'):
    assert len(data.shape) == 2, '只能针对2D数据进行pad！'
    assert type in ['in', 'out'], '只支持向内和向外pad，参数为in和out！'
    data_gen = data.copy()
    h, w = data.shape
    for i in range(h):
        for j in range(w):
            if data[i, j] == when:
                pad_value = value or data[i, j]
                for x, y in _get_coordinates(i, j, h, w, 1 if type == 'in' else -1):
                    data_gen[x, y] = pad_value
    return data_gen


def connect(data: np.ndarray, when: Union[float, int] = 1, stride: int = 1, mode=FILL_MODE_CHUNK):
    assert len(data.shape) == 2, '只能针对2D数据进行pad！'
    assert mode.upper() in ('CHUNK', 'LINE'), '目前只支持块填充和线填充'
    data_gen = data.copy()
    h, w = data.shape
    for i in range(h):
        for j in range(w):
            if data[i, j] == when:
                data_slice = data[i: i + stride + 1, j: j + stride + 1]
                coors = np.where(data_slice == when)
                if len(coors[0]) > 1:
                    data_gen_slice = data_gen[i: i + max(coors[0]) + 1, j: j + max(coors[1]) + 1]
                    data_slice = data[i: i + max(coors[0]) + 1, j: j + max(coors[1]) + 1]
                    if mode.upper() == 'CHUNK':
                        data_gen_slice[data_slice == 0] = when
                    else:
                        sorted_coors = sorted(list(zip(coors[0].tolist(), coors[1].tolist())))
                        final2fill = [sorted_coors[0]]
                        for idx in range(1, len(coors[0])):
                            fx, fy = sorted_coors[idx - 1]
                            lx, ly = sorted_coors[idx]
                            if lx >= fx and ly >= fy:
                                final2fill.append((lx, ly))
                        # print(final2fill)
                        for idx in range(1, len(final2fill)):
                            fx, fy = final2fill[idx - 1]
                            lx, ly = final2fill[idx]
                            data_gen_slice[fx:fx + 1, fy:ly + 1][data_slice[fx:fx + 1, fy:ly + 1] == 0] = when
                            data_gen_slice[fx:lx + 1, ly:ly + 1][data_slice[fx:lx + 1, ly:ly + 1] == 0] = when
    return data_gen


def fill_cross(data: np.ndarray, when: Union[float, int] = 1):
    coors = np.where(data == when)
    coors = list(zip(coors[0].tolist(), coors[1].tolist()))
    for x, y in coors:
        if (x - 1, y - 1) in coors and (x - 1, y) not in coors and (x, y - 1) not in coors:
            data[x - 1:x + 1, y - 1:y + 1] = when
        if (x + 1, y - 1) in coors and (x + 1, y) not in coors and (x, y - 1) not in coors:
            data[x:x + 2, y - 1:y + 1] = when
        if (x - 1, y + 1) in coors and (x - 1, y) not in coors and (x, y + 1) not in coors:
            data[x - 1:x + 1, y:y + 2] = when
        if (x + 1, y + 1) in coors and (x + 1, y) not in coors and (x, y + 1) not in coors:
            data[x:x + 2, y:y + 2] = when
    return data


def fill_hole(im_in):
    # 复制 im_in 图像
    im_floodfill = im_in.copy()
    # Mask 用于 floodFill，官方要求长宽+2
    h, w = im_in.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # floodFill函数中的seedPoint必须是背景
    isbreak = False
    seedPoint = (0, 0)
    for i in range(im_floodfill.shape[0]):
        for j in range(im_floodfill.shape[1]):
            if im_floodfill[i][j] == 0:
                seedPoint = (i, j)
                isbreak = True
                break
        if isbreak:
            break
    # 得到im_floodfill
    cv2.floodFill(im_floodfill, mask, seedPoint, 1)

    # 得到im_floodfill的逆im_floodfill_inv
    im_floodfill_inv = np.logical_not(im_floodfill)
    # 把im_in、im_floodfill_inv这两幅图像结合起来得到前景
    im_out = np.logical_or(im_in, im_floodfill_inv)

    # 保存结果
    return im_out


def canny_edges_3d(grayImage, pad_func):
    MIN_CANNY_THRESHOLD = 0
    MAX_CANNY_THRESHOLD = 1

    dim = np.shape(grayImage)

    edges_x = np.zeros(grayImage.shape, dtype=bool)
    edges_y = np.zeros(grayImage.shape, dtype=bool)
    edges_z = np.zeros(grayImage.shape, dtype=bool)
    edges = np.zeros(grayImage.shape, dtype=bool)

    # print(np.shape(edges))
    if pad_func is None:
        pad_func = lambda x: x

    for i in range(dim[0]):
        edges_x[i, :, :] = pad_func(canny(grayImage[i, :, :],
                                          low_threshold=MIN_CANNY_THRESHOLD,
                                          high_threshold=MAX_CANNY_THRESHOLD, sigma=0))
    for j in range(dim[1]):
        edges_y[:, j, :] = pad_func(canny(grayImage[:, j, :],
                                          low_threshold=MIN_CANNY_THRESHOLD,
                                          high_threshold=MAX_CANNY_THRESHOLD, sigma=0))

    for k in range(dim[2]):
        edges_z[:, :, k] = pad_func(canny(grayImage[:, :, k],
                                          low_threshold=MIN_CANNY_THRESHOLD,
                                          high_threshold=MAX_CANNY_THRESHOLD, sigma=0))

    # edges = canny(grayImage, low_threshold=MIN_CANNY_THRESHOLD, high_threshold=MAX_CANNY_THRESHOLD)
    for i in range(dim[0]):
        for j in range(dim[1]):
            for k in range(dim[2]):
                edges[i, j, k] = (edges_x[i, j, k] and edges_y[i, j, k]) or (edges_x[i, j, k] and edges_z[i, j, k]) or (
                        edges_y[i, j, k] and edges_z[i, j, k])
                # edges[i,j,k] = (edges_x[i,j,k]) or (edges_y[i,j,k]) or (edges_z[i,j,k])

    return edges


if __name__ == '__main__':
    def load_nii(filename):
        Image = nib.load(filename)
        img_arr = Image.get_fdata()
        name = filename.split('/')[-1]
        return img_arr.astype(np.float32), img_arr.shape, name, Image.affine, Image.header


    def save_nii(data, save_name, affine, header):
        new_img = nib.Nifti1Image(data.astype(np.int16), affine, header)
        nib.save(new_img, save_name)


    img, shape, name, affine, header = load_nii(
        "/Users/zhangzhiwei/Downloads/TrainingData/12m/012m_044a.v2_outskull_mask.nii.gz")
    edges = canny_edges_3d(img)
    save_nii(edges, "edge.nii.gz", affine, header)
