import itertools
import sys
import mmcv
import numpy as np
import copy

from terminaltables import AsciiTable
from time import time, clock
from tqdm import tqdm


JOINTS = ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder",
    "left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee", "left_ankle","right_ankle"]

def visibility_cls_eval(preds, gt_labels):
    """
    Use PCK with threshold of .5 of normalized distance (presumably head size)

    preds_v: 需要取pred的最后一维 ， pred shape: batch_size x  1 x 17 x 3

    coco:
    "nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder",
    "left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee", "left_ankle","right_ankle"

    """

    assert len(preds)==len(gt_labels)

    num_class = 3
    num_keypoints = 17

    prec_point = np.zeros((num_keypoints, num_class))

    count_point_tp = copy.deepcopy(prec_point)
    count_point_pd = copy.deepcopy(prec_point)
    count_point_gt = copy.deepcopy(prec_point)


    for pred, vlabel in tqdm(zip(preds, gt_labels)):

        pred_np = pred[0].cpu().numpy()
        pred_v = pred_np[:, 2]

        for j in range(num_keypoints):

            p = int(pred_v[j])
            v = int(vlabel[j])

            # print(j, p, v)

            count_point_pd[j, p] += 1
            count_point_gt[j, v] += 1

            if v==p:
                count_point_tp[j, v] += 1


    header = ['id', 'joint', 'prec(v=0)', 'prec(v=1)', 'prec(v=2)', 'aprec']
    table_data = [header]

    for k in range(num_keypoints):
        row_data = [k, JOINTS[k],
                    round(count_point_tp[k, 0] / count_point_pd[k, 0],3),
                    round(count_point_tp[k, 1] / count_point_pd[k, 1],3),
                    round(count_point_tp[k, 2] / count_point_pd[k, 2],3),
                    round(np.sum(count_point_tp, axis=1)[k] / np.sum(count_point_pd, axis=1)[k],3)
                    # round(np.sum(count_point_tp, axis=1)[k] / np.sum(count_point_gt, axis=1)[k],3)
                    ]
        table_data.append(row_data)

    # 暂时总评价是采用总tp数/总pd数（即总gt数）， 而不是采用多类求平均
    table_data.append(['', 'total',
                       round(np.sum(count_point_tp, axis=0)[0] / np.sum(count_point_pd, axis=0)[0],3),
                       round(np.sum(count_point_tp, axis=0)[1] / np.sum(count_point_pd, axis=0)[1],3),
                       round(np.sum(count_point_tp, axis=0)[2] / np.sum(count_point_pd, axis=0)[2],3),
                       round(np.sum(count_point_tp) / np.sum(count_point_pd),3)
                       # round(np.sum(count_point_tp) / np.sum(count_point_gt),3)
                       ])
    table = AsciiTable(table_data)
    table.inner_footing_row_border = True
    print(table.table)
