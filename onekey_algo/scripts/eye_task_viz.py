# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2022/10/25
# Home: www.medai.icu
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2022 All Rights Reserved.
import json
import os
import re

import numpy as np
from PIL import Image
from PIL.Image import Resampling

from onekey_algo.scripts.TaskViz import SegmentationVisualize, ClassificationVisualize


def gen_quadrant_mask(img_shape):
    s_h, s_w = img_shape
    s_h = s_h // 2
    s_w = s_w // 2
    corr = [[0, 0], [0, s_w], [s_h, 0], [s_h, s_w]]

    masks = []
    for x, y in corr:
        mask = np.zeros(img_shape)
        mask[x: x + s_h, y: y + s_w] = 1
        masks.append(mask)
    return masks


def intersection(img1, img2, img1_id: int = 1, img2_id: int = 1):
    assert img1.shape == img2.shape
    area = np.sum(np.logical_and(img1 == img1_id, img2 == img2_id))
    ratio = area / np.sum(img1 == img1_id)
    return int(area), float(ratio)


def getheight(x, img_id=1):
    bool_mask = np.where(x == img_id)[0]
    return max(bool_mask) - min(bool_mask) + 1


def ratio2level(x):
    if x <= 0.25:
        return 1
    elif x <= 0.50:
        return 2
    elif x <= 0.75:
        return 3
    else:
        return 4


def get_mask_file(img_file):
    bn, _ = os.path.splitext(os.path.basename(img_file))
    return os.path.join(os.path.join(os.path.dirname(img_file), 'masks', f"{bn}.png"))


class EyeTaskViz(object):
    def __init__(self, model_root: str, save_dir: str):
        self.bingbian_seg = SegmentationVisualize(os.path.join(model_root, 'bingbian'),
                                                  save_dir=os.path.join(save_dir, 'bingbian'))
        self.all_seg = SegmentationVisualize(os.path.join(model_root, 'all'),
                                             save_dir=os.path.join(save_dir, 'all'))
        self.clf = ClassificationVisualize(os.path.join(model_root, 'clf'), save_dir=os.path.join(save_dir, 'clf'))

    def predict(self, sample, model_name):
        bingbian_seg_name, all_seg_name = self.get_model_name(model_name)
        all_seg_results = self.all_seg.predict(sample, all_seg_name)
        bingbian_seg_results = self.bingbian_seg.predict(sample, bingbian_seg_name)
        clf_result = [None] * len(sample)
        if "corneal_scarring" in model_name:
            clf_result = self.clf.predict(sample, list(self.clf.model.keys())[0])
            clf_result = [re.split('[:|,]', res_['desc'])[1] for res_ in clf_result]
        post_results = self.post_process(model_name, all_seg_results, bingbian_seg_results)
        return [{"img": res_['img'],
                 'desc': json.dumps(post_res_, ensure_ascii=False) + (f', 分类：{clf_res_}' if clf_res_ else '')}
                for res_, post_res_, clf_res_ in zip(all_seg_results, post_results, clf_result)]

    def get_model_name(self, model_name):
        bingbian_seg_name = None
        all_seg_name = None
        for mn in self.bingbian_seg.model:
            if model_name in mn:
                bingbian_seg_name = mn
        for mn in self.all_seg.model:
            if model_name in mn:
                all_seg_name = mn
        # print(bingbian_seg_name, all_seg_name)
        assert bingbian_seg_name is not None and all_seg_name is not None, f"{model_name}并不是所有模型都加载！"
        return bingbian_seg_name, all_seg_name

    def viz_spec(self):
        return [{'model_name': f"corneal_scarring", "result_list": []},
                {'model_name': f"the_posterior_elastic_lamina", "result_list": []},
                {'model_name': f"corneal_ulcers", "result_list": []},
                {'model_name': f"anterior_chamber_empyema", "result_list": []},
                {'model_name': f"corneal_neovascularization", "result_list": []}]

    @staticmethod
    def post_process(model_name, all_seg_results, bingbian_seg_results):
        results = []
        gts = [get_mask_file(r['img']) for r in all_seg_results]
        preds = [get_mask_file(r['img']) for r in bingbian_seg_results]
        if "corneal_scarring" in model_name:
            for gt, pred in zip(gts, preds):
                gt_img = Image.open(gt)
                pred_a = np.array(Image.open(pred).resize(gt_img.size, resample=Resampling.NEAREST))
                gt_a = np.array(gt_img)
                quadrant_masks = gen_quadrant_mask(gt_a.shape)
                _, cornea_ratio = intersection(gt_a, pred_a, img1_id=1)
                _, pupil_ratio = intersection(gt_a, pred_a, img1_id=3)
                #     print(gt_a.shape, [(np.min(m_), np.max(m_) + 1) for m_ in np.where(pred_a==1)],
                #          [(np.min(m_), np.max(m_) + 1) for m_ in np.where(gt_a==1)])
                intersection_num = 0
                for mask in quadrant_masks:
                    _, ar = intersection(mask, pred_a)
                    if ar > 0.02:
                        intersection_num += 1
                results.append({"Percentage grading of corneal scar area": ratio2level(cornea_ratio),
                                "Corneal scarring involves the pupil": 'Yes' if pupil_ratio > 0.02 else 'No',
                                "Quadrant grading of corneal scarring": intersection_num})
        if "the_posterior_elastic_lamina" in model_name:
            for gt, pred in zip(gts, preds):
                gt_img = Image.open(gt)
                pred_a = np.array(Image.open(pred).resize(gt_img.size, resample=Resampling.NEAREST))
                gt_a = np.array(gt_img)
                print(gt_a.shape)
                quadrant_masks = gen_quadrant_mask(gt_a.shape)
                _, area_ratio = intersection(gt_a, pred_a, img1_id=1)
                ill_area = np.sum(pred_a == 1)
                #     print(gt_a.shape, [(np.min(m_), np.max(m_) + 1) for m_ in np.where(pred_a==1)],
                #          [(np.min(m_), np.max(m_) + 1) for m_ in np.where(gt_a==1)])
                #     for mask in quadrant_masks:
                #         result.extend(intersection(mask, pred_a))
                results.append({"Percentage grading of the posterior elastic lamina area": area_ratio})
        if "corneal_ulcers" in model_name:
            for gt, pred in zip(gts, preds):
                gt_img = Image.open(gt)
                pred_a = np.array(Image.open(pred).resize(gt_img.size, resample=Resampling.NEAREST))
                gt_a = np.array(gt_img)
                quadrant_masks = gen_quadrant_mask(gt_a.shape)
                _, cornea_ratio = intersection(gt_a, pred_a, img1_id=1)
                _, pupil_ratio = intersection(gt_a, pred_a, img1_id=3)
                #     print(gt_a.shape, [(np.min(m_), np.max(m_) + 1) for m_ in np.where(pred_a==1)],
                #          [(np.min(m_), np.max(m_) + 1) for m_ in np.where(gt_a==1)])
                intersection_num = 0
                for mask in quadrant_masks:
                    _, ar = intersection(mask, pred_a)
                    if ar > 0.02:
                        intersection_num += 1
                results.append({"Percentage grading of corneal ulcer area": ratio2level(cornea_ratio),
                                'Corneal ulcer involving pupil': 'Yes' if pupil_ratio > 0.02 else 'No',
                                "Quadrant grading of corneal ulcer area": intersection_num})
        if "anterior_chamber_empyema" in model_name:
            for gt, pred in zip(gts, preds):
                gt_img = Image.open(gt)
                pred_a = np.array(Image.open(pred).resize(gt_img.size, resample=Resampling.NEAREST))
                gt_a = np.array(gt_img)
                cornea_height = getheight(gt_a, img_id=1)
                ill_height = getheight(pred_a, img_id=1)
                #     print(gt_a.shape, [(np.min(m_), np.max(m_) + 1) for m_ in np.where(pred_a==1)],
                #          [(np.min(m_), np.max(m_) + 1) for m_ in np.where(gt_a==1)])
                results.append(
                    {"Percentile classification of anterior chamber empyema": ratio2level(ill_height / cornea_height)})
        if "corneal_neovascularization" in model_name:
            for gt, pred in zip(gts, preds):
                gt_img = Image.open(gt)
                pred_a = np.array(Image.open(pred).resize(gt_img.size, resample=Resampling.NEAREST))
                gt_a = np.array(gt_img)
                quadrant_masks = gen_quadrant_mask(gt_a.shape)
                _, cornea_ratio = intersection(gt_a, pred_a, img1_id=1)
                _, pupil_ratio = intersection(gt_a, pred_a, img1_id=3)
                #     print(gt_a.shape, [(np.min(m_), np.max(m_) + 1) for m_ in np.where(pred_a==1)],
                #          [(np.min(m_), np.max(m_) + 1) for m_ in np.where(gt_a==1)])
                intersection_num = 0
                for mask in quadrant_masks:
                    _, ar = intersection(mask, pred_a)
                    if ar > 0.02:
                        intersection_num += 1
                results.append({"Percentile classification of neovascularization area": ratio2level(cornea_ratio),
                                "Corneal neovascularization involves the pupil": 'Yes' if pupil_ratio > 0.02 else 'No',
                                "Quadrant grading of corneal neovascularization": intersection_num})
        return results


if __name__ == '__main__':
    eyev = EyeTaskViz(r'D:\20220423-Eye\models', r'D:\20220423-Eye\models/viz')
    results = eyev.predict(r'D:\20220423-Eye\角膜瘢痕\验证集\角膜瘢痕原始照片/1.jpg', model_name='蓝光下溃疡')
    # results = EyeTaskViz.post_process('角膜瘢痕', [], [])
    print(results)
    # print(json.dumps(results[0], ensure_ascii=False))
