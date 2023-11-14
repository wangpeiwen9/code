# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2022/08/01
# Blog: www.medai.icu
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2022 All Rights Reserved.

from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from onekey_algo.utils.about_log import logger


class MyDataset(Dataset):
    def __init__(self, data_path, not_use_columns: Union[str, List[str]] = None,
                 split='all', train_ratio: float = 0.8, transform=None, dropna: bool = True):
        self.data_path = data_path
        self.data = pd.read_csv(data_path)
        if dropna:
            ori_dim = self.data.shape[1]
            self.data = self.data.dropna(axis=1)
            logger.info(f"一共drop{ori_dim - self.data.shape[1]}列特征，最终保留{self.data.shape[1]}列。")
        else:
            self.data = self.data.fillna(0)

        if split == 'train':
            self.data = self.data[:int(len(self) * train_ratio)]
        elif split == 'val':
            self.data = self.data[int(len(self) * train_ratio):]
        elif split != 'all':
            raise ValueError(f'{split} not found!')
        if not_use_columns is not None:
            if isinstance(not_use_columns, str):
                not_use_columns = [not_use_columns]
            self.not_used = self.data[not_use_columns]
            # print(not_use_columns)
            for nc in not_use_columns:
                self.data.pop(nc)
        else:
            self.not_used = None
        # print(self.data.columns)
        self.data = np.array(self.data)
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    @property
    def feat_len(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        # if self.transform is not None:
        #     return self.transform(np.reshape(self.data[idx], (1, -1)))
        return torch.Tensor(np.reshape(self.data[idx], (1, -1)))


class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module 

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
            self,
            data_path: str,
            train_batch_size: int = 8,
            val_batch_size: int = 8,
            num_workers: int = 0,
            pin_memory: bool = False,
            **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None
        self.kwargs = kwargs

    def setup(self, stage: Optional[str] = None) -> None:
        train_transforms = transforms.Compose([transforms.ToTensor(), ])
        val_transforms = transforms.Compose([transforms.ToTensor(), ])

        self.train_dataset = MyDataset(self.data_dir, split='train', transform=train_transforms, **self.kwargs)
        # Replace CelebA with your dataset
        self.val_dataset = MyDataset(self.data_dir, split='val', transform=val_transforms, **self.kwargs)
        # Replace CelebA with your dataset
        self.test_dataset = MyDataset(self.data_dir, split='all', transform=val_transforms, **self.kwargs)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )


if __name__ == "__main__":
    ds = MyDataset('/Users/zhangzhiwei/Downloads/TCGA-RPPA-pancan-clean_norm.xena.csv', not_use_columns='ID',
                   transform=transforms.Compose([transforms.ToTensor(), ]))
    print(ds.not_used)
    for d in ds:
        print(d.shape)
        break
