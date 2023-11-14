# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2021/4/26
# Home: www.medai.icu
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2021 All Rights Reserved.
import copy
import os

import numpy as np
import torch
from PIL import Image

__all__ = ['PennFudanDataset', 'create_penn_dataset']


class PennFudanDataset(object):
    def __init__(self, root, transforms, task='segmentation'):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))
        self.num_classes = 2
        assert task in ['segmentation', 'detection']
        self.task = task

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB, because each color corresponds to a different instance
        # with 0 being background
        instance_target = Image.open(mask_path)
        mask = np.array(instance_target)
        if self.task == 'segmentation':
            map2semantic_array = copy.deepcopy(mask)
            map2semantic_array[mask > 0] = 1
            semantic_target = Image.fromarray(map2semantic_array)
            if self.transforms is not None:
                img, semantic_target = self.transforms(img, semantic_target)
            return img, semantic_target
        elif self.task == 'detection':
            # instances are encoded as different colors
            obj_ids = np.unique(mask)
            # first id is the background, so remove it
            obj_ids = obj_ids[1:]
            # split the color-encoded mask into a set of binary masks
            masks = mask == obj_ids[:, None, None]

            # get bounding box coordinates for each mask
            self.num_objs = len(obj_ids)
            boxes = []
            for i in range(self.num_objs):
                pos = np.where(masks[i])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                boxes.append([xmin, ymin, xmax, ymax])
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # there is only one class
            labels = torch.ones((self.num_objs,), dtype=torch.int64)

            masks = torch.as_tensor(masks, dtype=torch.uint8)
            image_id = torch.tensor([idx])
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            # suppose all instances are not crowd
            iscrowd = torch.zeros((self.num_objs,), dtype=torch.int64)
            target = {"boxes": boxes, "labels": labels, "masks": masks, "image_id": image_id, "area": area,
                      "iscrowd": iscrowd}
            if self.transforms is not None:
                img, target = self.transforms(img, target)
            return img, target
        else:
            raise ValueError(f'task {self.task} is not found. Only detection and segmentation supported!')

    def __len__(self):
        return len(self.imgs)

    def __repr__(self):
        head = f"Dataset is PennFudanDataset for task {self.task}."
        body = ["Samples: {}, Classes: {}".format(self.__len__(), self.num_classes)]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        lines = [head] + [f"\t{line}" for line in body]
        return '\n'.join(lines)


def create_penn_dataset(root, transform, task='segmentation'):
    """
    Create PennFudan dataset.
    Args:
        root: data source.
        transform: data transform.
        task: Task to use! semantic_segmentation|

    Returns: dataset
    """
    return PennFudanDataset(root, transform, task=task)
