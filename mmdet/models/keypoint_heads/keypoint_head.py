from __future__ import division

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from mmdet.core import (AnchorGenerator, anchor_target, delta2bbox, force_fp32,
                        multi_apply, multiclass_nms)
from ..builder import build_loss
from ..losses import smooth_l1_loss, smooth_l1_loss_ohkm, mse_loss_ohkm, mse_loss
from ..registry import HEADS


@HEADS.register_module
class KeypointHead(nn.Module):
    def __init__(self,
                 num_classes,
                 num_points,
                 # in_channels,
                 num_fcs=2,
                 out_channels_fc=1024,
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
        self.num_points = num_points
        self.num_fcs = num_fcs
        self.out_channels_fc = out_channels_fc
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.target_means = target_means
        self.target_stds = target_stds
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes - 1
        else:
            self.cls_out_channels = num_classes
        self.relu = nn.ReLU(inplace=True)

    def build_fc(self, num_fc_pre):
        self.cls_fcs = nn.ModuleList()
        self.reg_fcs = nn.ModuleList()
        out_fc_last = self.out_channels_fc
        for i in range(self.num_fcs-1):
            if i==0:
                self.cls_fcs.append(nn.Linear(num_fc_pre, self.out_channels_fc))
                self.reg_fcs.append(nn.Linear(num_fc_pre, self.out_channels_fc))
                out_fc_last = self.out_channels_fc
            else:
                self.cls_fcs.append(nn.Linear(out_fc_last, self.out_channels_fc))
                self.reg_fcs.append(nn.Linear(out_fc_last, self.out_channels_fc))
                out_fc_last = self.out_channels_fc
        self.fc_cls = nn.Linear(out_fc_last, self.num_points*self.num_classes)
        self.fc_reg = nn.Linear(out_fc_last, self.num_points*2)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x_cls = x
        cls_preds = []
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))
        cls_pred = self.fc_cls(x_cls)
        cls_preds.append(cls_pred)

        x_reg = x
        reg_preds = []
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))
        reg_pred = self.fc_reg(x_reg)
        reg_preds.append(reg_pred)
        return cls_preds, reg_preds

    def loss_single(self, cls_score, point_pred, labels, label_weights,
                    point_targets, point_weights, num_points, num_total_cls, num_total_points, batch_size, cfg):

        loss_cls_all = F.cross_entropy(
            cls_score, labels, reduction='none') * label_weights

        loss_cls = loss_cls_all.sum()/ num_total_cls

        loss_point = smooth_l1_loss(
            point_pred,
            point_targets,
            point_weights,
            beta=cfg.smoothl1_beta,
            avg_factor=num_total_points)

        # loss_point = mse_loss(
        #     point_pred,
        #     point_targets,
        #     point_weights,
        #     avg_factor=num_total_points)

        # mse loss会直接爆炸
        # loss_point = mse_loss_ohkm(
        #     point_pred,
        #     point_targets,
        #     point_weights,
        #     top_k=8)
        #     # avg_factor=num_total_points)

        # loss_point_all = smooth_l1_loss_ohkm(
        #     point_pred,
        #     point_targets,
        #     point_weights,
        #     num_points,
        #     top_k=6)
        # loss_point = loss_point_all/batch_size

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

        #num_total_cls = torch.sum(all_label_weights)
        num_total_cls = torch.sum(all_label_weights)/self.num_points*2
        num_total_points = torch.sum(all_label_weights_tmp>0)
        num_points = torch.sum(all_label_weights_tmp > 0, dim=1)

        (losses_cls, losses_point) = multi_apply(
            self.loss_single,
            all_cls_scores,
            all_point_preds,
            all_labels,
            all_label_weights,
            all_point_targets,
            all_point_weights,
            num_points,
            num_total_cls=num_total_cls,
            num_total_points=num_total_points,
            batch_size=num_points.shape[0],
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

    def get_points(self, cls_scores, point_preds, img_metas, cfg,
                   rescale=False):
        point_list = []
        for img_id in range(len(img_metas)):
            cls_scores_one = cls_scores[img_id].reshape( -1, self.cls_out_channels)
            point_preds_one = point_preds[img_id].reshape( -1, 2)
            assert cls_scores_one.size()[0]==point_preds_one.size()[0]
            if self.use_sigmoid_cls:
                scores = cls_scores_one.sigmoid()
                max_scores, max_idxs = scores.max(dim=1)
            else:
                scores = cls_scores_one.softmax(-1)
                max_scores, max_idxs = scores.max(dim=1)
            points_decoded = self.decode_delta_single(point_preds_one,img_metas[img_id])
            max_idxs = max_idxs.reshape(max_idxs.size()[-1],1)
            point_list.append(torch.cat((points_decoded,max_idxs.float()),1))
        return  point_list

    def decode_delta_single(self, deltas, img_meta):
        points_tmp = deltas.float()
        target_means = [self.target_means[0], self.target_means[0]]
        target_stds = [self.target_stds[0], self.target_stds[0]]
        target_means = points_tmp.new_tensor(target_means).unsqueeze(0)
        target_stds = points_tmp.new_tensor(target_stds).unsqueeze(0)
        points_tmp2 = points_tmp.mul_(target_stds).add_(target_means)
        decoded_x = points_tmp2[:,0] * img_meta['ori_shape'][1]
        decoded_y = points_tmp2[:,1] * img_meta['ori_shape'][0]
        return  torch.stack([decoded_x, decoded_y], -1)


