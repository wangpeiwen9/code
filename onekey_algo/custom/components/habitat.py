# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2023/10/26
# Home: www.medai.icu
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2023 All Rights Reserved.
import os

import numpy as np
import pandas as pd
import pyecharts.options as opts
from pyecharts.charts import Scatter3D
from scipy import spatial
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

from onekey_algo.custom.components.comp1 import compress_df_feature


def map2_anchor_feature(features, anchor_features):
    # print(features[0]['ID'])
    total_habitats = len(features)
    mapping = [None] * len(features)
    sim_matrix = np.zeros((total_habitats, total_habitats))
    for idx, feature in enumerate(features):
        for jdx, anchor_feature in enumerate(anchor_features):
            cos_sim = 1 - spatial.distance.cosine(feature[[c for c in feature.columns if c != 'ID']],
                                                  anchor_feature[[c for c in anchor_feature.columns if c != 'ID']])
            sim_matrix[idx][jdx] = cos_sim
    # sim_matrix = np.reshape(sim_matrix, -1)
    for idx in range(total_habitats):
        max_idx = np.argmax(sim_matrix)
        i = max_idx // total_habitats
        j = max_idx % total_habitats
        mapping[i] = j
        sim_matrix[i, :] = -1
        sim_matrix[:, j] = -1
    assert None not in mapping, f"映射逻辑出错！！"
    return mapping


def habitat_feature_fusion(*rad_features, mode='raw', anchor=None, with_remapping: bool = False, n_neighbors=5):
    """
    解决生境分析最后一公里的特征融合问题。 针对所有样本放在一起聚类，支持3种不同的特征融合方法
       0. raw, 直接取交集。
       1. 特征合并
          1. max, 所有具有相同特征名称的生境区域，保留最大值。
          2. min, 所有具有相同特征名称的生境区域，保留最小值。
          3. mean, 所有具有相同特征名称的生境区域，保留均值。
       2. fill,基于KNN方法，对所有的缺失值进行填充，与MICE方法类似。
       3. remap，根据一个锚点，根据相似度进行计算，重新映射这些生境区域。
    Args:
        *rad_features: 待融合的生境特征
        mode: 融合方法
        anchor: 锚点ID，当remap模式是生效，
        with_remapping: remap模式是，是否返回映射表。
        n_neighbors: fill模式下，使用的k近邻的数量。

    Returns: 融合之后的特征

    """
    if mode == 'raw':
        rad = rad_features[0]
        for r in rad_features[1:]:
            rad = pd.merge(rad, r, on='ID', how='inner')
        return rad
    elif mode == 'fill':
        ids = set()
        for r in rad_features:
            ids |= set(r['ID'])
        rad = pd.DataFrame(sorted(ids), columns=['ID'])
        for r in rad_features:
            rad = pd.merge(rad, r, on='ID', how='left')
        imputer = KNNImputer(n_neighbors=n_neighbors)
        rad_none_id = rad[[c for c in rad.columns if c != 'ID']]
        rad_none_id = pd.DataFrame(imputer.fit_transform(rad_none_id), columns=rad_none_id.columns)
        rad = pd.concat([rad['ID'], rad_none_id], axis=1)
        return rad
    elif mode == 'remap':
        cmp_features = []
        for data in rad_features:
            scaler = MinMaxScaler()
            # 对 DataFrame 中的每一列执行0-1标准化
            data_scaled = scaler.fit_transform(data[[c for c in data.columns if c != 'ID']])
            # 将标准化的数据转回DataFrame
            data_scaled = pd.DataFrame(data_scaled, columns=[c for c in data.columns if c != 'ID'])
            cmp_features.append(pd.concat([data['ID'], data_scaled], axis=1))
        ids = rad_features[0]['ID']
        if anchor is None:
            anchor = ids[0]
        anchor_features = [r[r['ID'] == anchor] for r in cmp_features]

        mapping_features = [[] for _ in range(len(rad_features))]
        mappings = []
        for sample in ids:
            mapping = map2_anchor_feature([r[r['ID'] == sample] for r in cmp_features], anchor_features)
            for idx, m in enumerate(mapping):
                mapping_features[m].append(rad_features[idx][rad_features[idx]['ID'] == sample])
            mappings.append(mapping)
        mapping = pd.concat([ids, pd.DataFrame(mappings)], axis=1)
        rad = habitat_feature_fusion(*[pd.DataFrame(np.concatenate(r, axis=0), columns=rad_features[idx].columns)
                                       for idx, r in enumerate(mapping_features)])
        if with_remapping:
            return rad, mapping
        else:
            return rad
    elif mode in ['max', 'min', 'mean']:
        rad_features = pd.DataFrame(np.concatenate(rad_features, axis=0), columns=rad_features[0].columns)
        rad = rad_features.groupby('ID').agg(mode).reset_index()
        return rad
    else:
        raise ValueError(f"您输入的聚合方法不支持: {mode}, {__doc__}")


def habitat_viz(data_npy, with_voxels: bool = False):
    s3d = Scatter3D()
    assert os.path.exists(data_npy), f'{data_npy}文件路径不存在！'
    data = np.load(data_npy)
    assert data.shape[
               1] >= 4, f"{data_npy}文件至少包括3个特征以及1个聚类类别，一共4列数据，现在数据只有{data.shape[1]}列。"
    clusters = data[:, -1]
    uni_habitats = np.unique(clusters)
    features = compress_df_feature(data[:, :-1], dim=3)
    for uni_habitat in sorted(uni_habitats):
        data_habitat = [[feat[0], feat[1], feat[2], clu, 1] for feat, clu in zip(np.array(features), clusters) if
                        clu == uni_habitat]
        voxel_str = f' {len(data_habitat)}({len(data_habitat) * 100 / data.shape[0]:0.2f}%) voxels' if with_voxels else ''
        s3d.add(series_name=f'Habitat {int(uni_habitat)}' + voxel_str,
                data=data_habitat, grid3d_opts=opts.Grid3DOpts(width=320, height=200, depth=200))

    s3d.set_global_opts(visualmap_opts=[
        opts.VisualMapOpts(
            is_show=False,
            type_="size",
            is_calculable=False,
            dimension=4,
            pos_bottom="10",
            max_=1,
            range_size=[2, 3],
        )
    ],
        legend_opts=opts.LegendOpts(is_show=True, orient='vertical', pos_bottom=0, pos_right=0, border_width=1),
        toolbox_opts=opts.ToolboxOpts(is_show=True,
                                      feature=opts.ToolBoxFeatureOpts(magic_type={},
                                                                      save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(
                                                                          background_color='white'))))

    return s3d.render_notebook()


if __name__ == '__main__':
    rad1 = pd.read_csv('rad_features_h1.csv')
    rad2 = pd.read_csv('rad_features_h2.csv')
    rad3 = pd.read_csv('rad_features_h3.csv')
    habitat_feature_fusion(rad1, rad2, rad3, mode='fill')
