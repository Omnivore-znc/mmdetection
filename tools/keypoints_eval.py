from argparse import ArgumentParser

import mmcv
import numpy as np

from mmdet import datasets
from mmdet.core import pck_eval, computeOks, visibility_cls_eval

from tqdm import tqdm


def keypoints_eval(result_files,
              result_types,
              dataset,
              # coco,
              max_dets=(100, 300, 1000),
              classwise=False):

    for res_type in result_types:
        assert res_type in [
            'human-points'
        ]

    eval_types = ['oks', 'pck']  # pck or oks

    det_results = mmcv.load(result_files)

    # num_img x num_obj_per_img x 17 x 3
    # print(len(det_results), len(det_results[0]), det_results[0][0].shape)
    # print(det_results)


    gt_points = []
    gt_labels = []
    gt_normalize = []
    gt_height = []
    gt_width = []
    # print(len(dataset))

    num_points = 0

    for i in tqdm(range(len(dataset))):
        # print(i)
        img = dataset.load_annotations
        ann = dataset.get_ann_info(i)
        points = ann['points']
        labels = ann['labels']
        gt_points.append(points)
        gt_labels.append(labels)

        for v in labels:
            if v==1 or v==2:
                num_points+=1


        if 'pck' in eval_types:
            normalize = np.sqrt(ann['height']**2+ann['width']**2)
            gt_normalize.append(normalize)
        if 'oks' in eval_types:
            gt_height.append(ann['height'])
            gt_width.append(ann['width'])

        # print(type(points))
        # print(ann)

    print('num points:', num_points)
    if 'pck' in eval_types:
        print('\nStarting evaluate PCK: \n')
        pck_eval(det_results, gt_points, gt_labels, gt_normalize)
    if 'oks' in eval_types:
        print('\nStarting evaluate OKS mAP: \n')
        computeOks(det_results, gt_points, gt_labels, gt_height, gt_width)

    print('\nStarting evaluate Visibility Presicion: \n')
    visibility_cls_eval(det_results, gt_labels)


def main():
    parser = ArgumentParser(description='VOC Evaluation')
    parser.add_argument('result', help='result file path')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--iou-thr',
        type=float,
        default=0.5,
        help='IoU threshold for evaluation')
    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    test_dataset = mmcv.runner.obj_from_dict(cfg.data.test, datasets)
    eval_types = ['pck', 'oks']
    keypoints_eval(args.result_file, eval_types, test_dataset)


if __name__ == '__main__':
    main()
