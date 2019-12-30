import os
import torch
import numpy as np
import os
import cv2
from ..registry import DETECTORS
from .single_stage import SingleStageDetector
from .. import builder

def model_visualize_keypoints(img,
                           img_metas,
                           gt_points,
                           gt_labels,
                           save_dir):
    for m in range(len(img_metas)):
        meta = img_metas[m]
        ydt = int((meta['img_shape'][0] - meta['img_resize_shape'][0]) / 2)
        xdt = int((meta['img_shape'][1] - meta['img_resize_shape'][1]) / 2)
        img_tmp = np.array(img[m].cpu().numpy().transpose(1, 2, 0).astype(np.uint8))
        points_tmp = gt_points[m].cpu().numpy().astype(np.int)
        labels_tmp = gt_labels[m].cpu().numpy().astype(np.int)
        for i in range(len(points_tmp)):
            center = points_tmp[i]
            if labels_tmp[i] == 1:
                img_tmp = cv2.circle(img_tmp, (xdt + center[0], ydt + center[1]), 3, (255, 0, 0), 2)
            elif labels_tmp[i] == 2 and i%2!=0:
                img_tmp = cv2.circle(img_tmp, (xdt + center[0], ydt + center[1]), 3, (0, 255, 0), 2)
            elif labels_tmp[i] == 2 and i%2==0:
                img_tmp = cv2.circle(img_tmp, (xdt + center[0], ydt + center[1]), 3, (0, 0, 255), 2)
        #save_dir = '/opt/space_host/zhongnanchang/mmdet_models/work_dirs/tmp_post'
        _, name = os.path.split(meta['filename'])
        save_path = os.path.join(save_dir, name)
        cv2.imwrite(save_path, img_tmp)

@DETECTORS.register_module
class PointBoxSingleStageDetector(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 point_head=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(PointBoxSingleStageDetector, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained)
        if point_head is not None:
            self.point_head = builder.build_head(point_head)
            input = torch.randn(1, 3, backbone.input_height, backbone.input_width)
            h = self.extract_feat(input)
            h = h.view(h.size(0), -1)
            print('BlazeFace ouput size = {}, size[1] = {}'.format(h.size(), h.size()[1]))
            self.point_head.build_fc(h.size()[1])

    def init_weights(self, pretrained=None):
        super(SingleStageDetector, self).init_weights(pretrained)
        if self.with_point:
            self.point_head.init_weights()

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmedetection/tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs_bb = None
        outs_pt = None
        if self.with_bbox:
            outs_bb = self.bbox_head(x)
        if self.with_point:
            outs_pt = self.point_head(x)
        return outs_bb,outs_pt

    def forward_train(self,
                      img,
                      img_metas,
                      gt_points,
                      gt_labels,
                      gt_bboxes=None,
                      gt_bboxes_ignore=None):

        # # todo
        # # print(sys._getframe())
        # # print(sys._getframe().f_lineno )
        # model_visualize_keypoints( img,
        #                               img_metas,
        #                               gt_points,
        #                               gt_labels,
        #                               '/opt/space_host/zhongnanchang/mmdet_models/work_dirs/tmp_post')

        x = self.extract_feat(img)
        if isinstance(x,tuple):
            x = x[0]
        losses_pt = None
        if self.with_bbox:
            # outs = self.bbox_head(x)
            # loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
            # losses = self.bbox_head.loss(
            #     *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            raise NotImplementedError
        if self.with_point:
            outs = self.point_head(x)
            loss_inputs = outs + (gt_points, gt_labels, img_metas, self.train_cfg)
            losses_pt = self.point_head.loss(*loss_inputs)
        return losses_pt

    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)
        outs_pt = None
        if self.with_bbox:
            raise NotImplementedError
        if self.with_point:
            outs_pt = self.point_head(x)
            point_inputs = outs_pt + (img_meta, self.test_cfg, rescale)
            point_list = self.point_head.get_points(*point_inputs)
        return point_list

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError

