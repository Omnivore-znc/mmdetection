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
                 target_means=(0.5, 0.5),
                 target_stds=(1.0, 1.0),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     #use_sigmoid=True,
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_point=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)):
        super(KeypointHead, self).__init__()
        #self.in_channels = in_channels
        self.num_classes = num_classes
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.target_means = target_means
        self.target_stds = target_stds
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes - 1
        else:
            self.cls_out_channels = num_classes

    def build_fc(self, num_fc_pre):
        self.fc_cls = nn.Linear(num_fc_pre, 51)
        self.fc_reg = nn.Linear(num_fc_pre, 34)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        #print('BlazeFace ouput size = {}'.format(h.size()))
        cls_preds = []
        reg_preds = []
        cls_pred = self.fc_cls(x)
        cls_preds.append(cls_pred)
        reg_pred = self.fc_reg(x)
        reg_preds.append(reg_pred)
        return cls_preds, reg_preds

    def loss_single(self, cls_score, point_pred, labels, label_weights,
                    point_targets, point_weights, num_total_cls, num_total_points, cfg):
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
        loss_cls = loss_cls_all.sum()/ num_total_cls

        loss_point = smooth_l1_loss(
            point_pred,
            point_targets,
            point_weights,
            beta=cfg.smoothl1_beta,
            avg_factor=num_total_points)
        return (loss_cls, loss_point)

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
            s.reshape(num_images, -1, self.cls_out_channels) for s in cls_scores
        ], 1)
        all_labels = torch.cat(gt_labels, -1).view(num_images, -1)
        all_label_weights = torch.tensor(all_labels)
        all_label_weights[...] = 1

        all_point_preds = torch.cat([
            b.reshape(num_images, -1, 2) for b in point_preds
        ], -2)
        all_point_targets = torch.cat(gt_points,
                                     -2).view(num_images, -1, 2)

        # get deltas
        (all_point_targets, all_point_targets_ori) = multi_apply(self.delta_single,
                                        all_point_targets,
                                        img_metas
                                        )
        all_point_targets = torch.cat(all_point_targets, -2).view(num_images, -1, 2)
        # point weight
        all_point_weights = torch.tensor(all_point_targets)
        all_label_weights_tmp = torch.tensor(all_labels)
        all_label_weights_tmp[all_label_weights_tmp > 0] = 1
        all_label_weights_tmp[all_label_weights_tmp < 0] = 0
        all_point_weights[all_label_weights_tmp==0,:] = 0
        all_point_weights[all_label_weights_tmp>0, :] = 1
        #all_point_weights =

        num_total_cls = torch.sum(all_label_weights)
        num_total_points = torch.sum(all_label_weights_tmp>0)
        (losses_cls, losses_point) = multi_apply(
            self.loss_single,
            all_cls_scores,
            all_point_preds,
            all_labels,
            all_label_weights,
            all_point_targets,
            all_point_weights,
            num_total_cls=num_total_cls,
            num_total_points=num_total_points,
            cfg=cfg)
        return dict(loss_cls=losses_cls, loss_point=losses_point)

    def delta_single(self, point_targets, img_meta):
        point_targets_tmp = point_targets.float()
        delta_x = point_targets_tmp[:,0]/img_meta['img_shape'][1]
        delta_y = point_targets_tmp[:,1]/img_meta['img_shape'][0]
        point_targets_tmp = torch.stack([delta_x, delta_y],-1)
        target_means = [self.target_means[0],self.target_means[0]]
        target_stds = [self.target_stds[0], self.target_stds[0]]
        target_means = point_targets_tmp.new_tensor(target_means).unsqueeze(0)
        target_stds =  point_targets_tmp.new_tensor(target_stds).unsqueeze(0)
        point_targets_tmp = point_targets_tmp.sub_(target_means).div_(target_stds)
        return  (point_targets_tmp,point_targets)

