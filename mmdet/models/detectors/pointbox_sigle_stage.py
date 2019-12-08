import sys
import numpy as np
import torch.nn as nn
import cv2
from ..registry import DETECTORS
from .single_stage import SingleStageDetector
from .. import builder

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
            self.point_head.build_fc(self.backbone.num_fc_pre)

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
        # # log
        # #img.
        # # todo-znc
        # print(sys._getframe())
        # #print(sys._getframe().f_lineno )
        # img_tmp =  np.array(img[0].cpu().numpy().transpose(1,2,0).astype(np.uint8))
        # for i in range(len(gt_points[0])):
        #     center = gt_points[0].cpu().numpy().astype(np.int)[i]
        #     cv2.circle(img_tmp,(center[0],center[1]),2,(0,255,0))
        # cv2.imwrite("haha2.jpg",img_tmp)
        #
        # img_tmp = np.ones((128,64,3),np.uint8)*200
        # for i in range(len(gt_points[0])):
        #     center = gt_points[0].cpu().numpy().astype(np.int)[i]
        #     cv2.circle(img_tmp,(center[0],center[1]),2,(0,255,0))
        # cv2.imwrite("haha3.jpg", img_tmp)

        x = self.extract_feat(img)
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
            # outs = self.bbox_head(x)
            # bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
            # bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
            # bbox_results = [
            #     bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            #     for det_bboxes, det_labels in bbox_list
            # ]
            raise NotImplementedError
        if self.with_point:
            outs_pt = self.point_head(x)
        return outs_pt

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError

