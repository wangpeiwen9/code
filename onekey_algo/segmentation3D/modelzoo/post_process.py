# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2023/06/14
# Forum: www.medai.icu
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2023 All Rights Reserved.
import SimpleITK as sitk
import numpy as np
from monai.config import KeysCollection
from monai.transforms import MapTransform


class RemoveSmallObjectsPerLabel(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False,
                 min_size=64, verbose: bool = False, force2: int = 0):
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.min_size = min_size
        self.verbose = verbose
        self.force2 = force2

    def __call__(self, data):
        for k in self.keys:
            batch_data = []
            for idx in range(data[k].shape[0]):
                sel_data = data[k][idx]
                print(sel_data.shape)
                sel_data = sitk.GetImageFromArray(sel_data)
                sitk.WriteImage(sel_data, f"{idx}.nii.gz")
                cc_filter = sitk.ConnectedComponentImageFilter()
                cc_filter.SetFullyConnected(True)
                omask_array = sitk.GetArrayFromImage(cc_filter.Execute(sel_data))
                unique_labels = np.unique(omask_array)
                mask_label_voxels = {}
                for ul in unique_labels:
                    mask_label_voxels[ul] = np.sum(omask_array == ul)
                mask_label_voxels = sorted(mask_label_voxels.items(), key=lambda x: x[1], reverse=True)
                mask_postprocess = np.ones_like(omask_array)
                for idx, (ul, cnt) in enumerate(mask_label_voxels):
                    if cnt < self.min_size:
                        mask_postprocess[omask_array == ul] = self.force2
                if self.verbose:
                    print(unique_labels, mask_label_voxels)
                batch_data.append(mask_postprocess * data[k])
            data[k] = np.array(batch_data)
        return data


if __name__ == '__main__':
    a = np.array([[[0, 2, 2, 0, 1], [0, 2, 2, 0, 1], [0, 0, 0, 0, 1], [1, 1, 0, 1, 1], [1, 1, 1, 1, 1]]])
    data = {'pred': a}
    print(RemoveSmallObjectsPerLabel(keys='pred', min_size=5, verbose=True)(data))
