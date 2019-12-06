from __future__ import division

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from mmdet.core import (AnchorGenerator, anchor_target, delta2bbox, force_fp32,
                        multi_apply, multiclass_nms)
from ..builder import build_loss
from ..losses import smooth_l1_loss
from ..registry import HEADS


@HEADS.register_module
class KeypointHead(nn.Module):
    def __init__(self,
                 num_classes,
                 # in_channels,
                 # feat_channels=256,
                 target_means=(.0, .0),
                 target_stds=(1.0, 1.0),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_point=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)):
        super(KeypointHead, self).__init__()
        #self.in_channels = in_channels
        self.num_classes = num_classes

    def build_fc(self, num_fc_pre):
        self.fc_cls = nn.Linear(num_fc_pre, 51)
        self.fc_reg = nn.Linear(num_fc_pre, 34)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        #print('BlazeFace ouput size = {}'.format(h.size()))
        cls_pred = self.fc_cls(x)
        reg_pred = self.fc_reg(x)
        return [cls_pred, reg_pred]

    def loss_single(self, cls_score, point_pred, labels, label_weights,
                    point_targets, point_weights, num_total_samples, cfg):
        loss_cls_all = F.cross_entropy(
            cls_score, labels, reduction='none') * label_weights
        # pos_inds = (labels > 0).nonzero().view(-1)
        # neg_inds = (labels == 0).nonzero().view(-1)
        #
        # num_pos_samples = pos_inds.size(0)
        # num_neg_samples = cfg.neg_pos_ratio * num_pos_samples
        # if num_neg_samples > neg_inds.size(0):
        #     num_neg_samples = neg_inds.size(0)
        # topk_loss_cls_neg, _ = loss_cls_all[neg_inds].topk(num_neg_samples)
        # loss_cls_pos = loss_cls_all[pos_inds].sum()
        # loss_cls_neg = topk_loss_cls_neg.sum()
        loss_cls = loss_cls_all.sum()/ num_total_samples

        loss_bbox = smooth_l1_loss(
            point_pred,
            point_targets,
            point_weights,
            beta=cfg.smoothl1_beta,
            avg_factor=num_total_samples)
        return loss_cls, loss_bbox

    def loss(self,
             cls_scores,
             point_preds,
             gt_points,
             gt_labels,
             img_metas,
             cfg):
        device = cls_scores[0].device

        num_images = len(img_metas)

        all_cls_scores = torch.cat([
            s.permute(0, 2, 3, 1).reshape(
                num_images, -1, self.cls_out_channels) for s in cls_scores
        ], 1)
        all_labels = torch.cat(gt_labels, -1).view(num_images, -1)
        all_label_weights = None

        all_point_preds = torch.cat([
            b.permute(0, 2, 3, 1).reshape(num_images, -1, 2)
            for b in point_preds
        ], -2)
        all_point_targets = torch.cat(gt_points,
                                     -2).view(num_images, -1, 2)
        all_point_weights = None

        num_total_pos = cfg.model.point_head.num_keypoints*num_images
        losses_cls, losses_point = multi_apply(
            self.loss_single,
            all_cls_scores,
            all_point_preds,
            all_labels,
            all_label_weights,
            all_point_targets,
            all_point_weights,
            num_total_samples=num_total_pos,
            cfg=cfg)
        return dict(loss_cls=losses_cls, loss_bbox=losses_point)
