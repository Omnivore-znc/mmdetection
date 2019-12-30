import os.path as osp

import json
import mmcv
import numpy as np

from .custom import CustomDataset
from .registry import DATASETS


@DATASETS.register_module
class JSONDataset(CustomDataset):

    def __init__(self, min_size=None, **kwargs):
        super(JSONDataset, self).__init__(**kwargs)
        self.cat2label = {cat: i + 1 for i, cat in enumerate(self.CLASSES)}
        self.min_size = min_size

    def load_annotations(self, ann_file):
        img_infos = []
        img_ids = mmcv.list_from_file(ann_file)
        for img_id in img_ids:
            filename = 'JPEGImages/{}.jpg'.format(img_id)
            ann_path = '{}/Annotations/{}.json'.format(self.img_prefix,img_id)

            with open(ann_path) as annfile:

                ann = json.load(annfile)

                img_w = ann['image_w']
                img_h = ann['image_h']
                assert img_w > 0 and img_w < 2000 and img_h > 0 and img_h < 2000


                # 暂时先按coco17点进行训练
                num_keypt = 17

                if num_keypt == 0:
                    raise "keypoint num invalid in file {}".format(ann_path)

                num_keypt_valid = ann['num_keypoints']

            img_infos.append(
                dict(id=img_id, filename=filename, width=img_w, height=img_h, num_keypt_valid=num_keypt_valid))
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        img_w = self.img_infos[idx]['width']
        img_h = self.img_infos[idx]['height']

        num_keypt_valid = self.img_infos[idx]['num_keypt_valid']
        num_keypt = 17

        json_path = osp.join(self.img_prefix, 'Annotations',
                            '{}.json'.format(img_id))
        points = []
        labels = []
        points_ignore = []
        labels_ignore = []
        with open(json_path) as annfile:

            # 改成json
            ann = json.load(annfile)
            points_v = ann['keypoints']

            for i in range(num_keypt):
                point = [int(points_v[3*i] + 0.5), int(points_v[3*i+1] + 0.5)]
                labels.append(int(points_v[3*i+2]))
                points.append(point)
            assert (len(points) == (num_keypt))
            assert (len(labels) == num_keypt)

        if not points:
            points = np.zeros((0,2))
            labels =  np.zeros((0,1))
        points = np.array(points, ndmin=2) - 1
        labels = np.array(labels)
        points_ignore = np.zeros((0,2))
        labels_ignore = np.zeros((0,))
        ann = dict(points = points.astype(np.float32),
                   labels = labels.astype(np.int64),
                   points_ignore = points_ignore,
                   labels_ignore = labels_ignore,
                   height = img_h,
                   width = img_w
                   )
        return ann

    def _filter_imgs(self, min_size=4):
        """Filter images too small."""
        valid_inds = []
        for i, img_info in enumerate(self.img_infos):
            if min(img_info['width'], img_info['height']) >= min_size and img_info['num_keypt_valid']>=2:
                valid_inds.append(i)
            #else:
                #print('not enough valid keypoints: {}'.format(img_info['filename']))
        return valid_inds