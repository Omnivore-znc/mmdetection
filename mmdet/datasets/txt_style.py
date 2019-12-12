import os.path as osp
import xml.etree.ElementTree as ET

import mmcv
import numpy as np

from .custom import CustomDataset
from .registry import DATASETS


@DATASETS.register_module
class TXTDataset(CustomDataset):

    def __init__(self, min_size=None, **kwargs):
        super(TXTDataset, self).__init__(**kwargs)
        self.cat2label = {cat: i + 1 for i, cat in enumerate(self.CLASSES)}
        self.min_size = min_size

    def load_annotations(self, ann_file):
        img_infos = []
        img_ids = mmcv.list_from_file(ann_file)
        for img_id in img_ids:
            filename = 'IMAGE_ANNOTATIONS/{}.jpg'.format(img_id)
            ann_path = '{}/IMAGE_ANNOTATIONS/{}.txt'.format(self.img_prefix,img_id)
            ## image width-height
            img_w = 0
            img_h = 0
            with open(ann_path) as annfile:
                # while 1:
                #     wh = annfile.readline()
                #     if len(wh.split(' ')) != 2:
                #         continue
                num_keypt = 0
                while 1:
                    num = annfile.readline().split(' ')
                    if len(num) == 2:
                        img_w = int(num[0])
                        img_h = int(num[1])
                        assert img_w > 0 and img_w < 2000 and img_h > 0 and img_h < 2000
                    if len(num) == 1:
                        num = num[0]
                        if int(num) > 0 and int(num) < 500:
                            num_keypt = int(num)
                            break
                if num_keypt == 0:
                    raise "keypoint num invalid in file {}".format(ann_path)
                num_keypt_valid = 0
                while 1:
                    num = annfile.readline()
                    if int(num) >= 0 and int(num) < 100:
                        num_keypt_valid = int(num)
                        break
            img_infos.append(
                dict(id=img_id, filename=filename, width=img_w, height=img_h, num_keypt_valid=num_keypt_valid))
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        txt_path = osp.join(self.img_prefix, 'IMAGE_ANNOTATIONS',
                            '{}.txt'.format(img_id))
        points = []
        labels = []
        points_ignore = []
        labels_ignore = []
        with open(txt_path) as annfile:
            num_keypt = 0
            while 1:
                num = annfile.readline().split(' ')
                if len(num) == 2:
                    continue
                if len(num)==1:
                    num = num[0]
                    if int(num)>0 and int(num) < 500:
                        num_keypt = int(num)
                        break
            if num_keypt == 0:
                raise "keypoint num invalid in file {}".format(txt_path)

            num_keypt_valid = 0
            while 1:
                num = annfile.readline()
                if int(num) >= 0 and int(num) < 100:
                    num_keypt_valid = int(num)
                    break

            #ann.append(num_keypt)
            #ann.append(num_keypt_valid)
            for i in range(num_keypt):
                while 1:
                    line = annfile.readline()
                    ele_arr = line.split(' ')
                    if len(ele_arr) == 3:
                        point = [int(float(ele_arr[0]) + 0.5), int(float(ele_arr[1]) + 0.5)]
                        labels.append(int(ele_arr[2]))
                        break
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
                   labels_ignore = labels_ignore)
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