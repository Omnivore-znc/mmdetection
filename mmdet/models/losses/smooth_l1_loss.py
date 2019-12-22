import torch
import torch.nn as nn

from ..registry import LOSSES
from .utils import weighted_loss


def ohkm(loss, num_points, top_k):
    # ohkm_loss = copy.deepcopy(loss)
    loss_point = loss.sum((1), keepdim=True)
    topk_val, topk_idx = torch.topk(loss_point, k=top_k, dim=0, sorted=False)

    tmp_loss = torch.gather(loss_point, 0, topk_idx)

    # min(top_k, num_vis)
    num_point = num_points if num_points<top_k else top_k
    ohkm_loss = torch.sum(tmp_loss) / num_point

    return ohkm_loss

def smooth_l1_loss_ohkm(pred, target, weights, num_points, top_k=17):
    assert pred.size() == target.size() and target.numel() > 0
    # diff = torch.abs(pred - target)

    loss = smooth_l1_loss(pred, target, reduction='none')
    loss_weight = loss*weights

    loss_ohkm = ohkm(loss_weight, num_points, top_k)

    return loss_ohkm


@weighted_loss
def smooth_l1_loss(pred, target, beta=1.0):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    return loss

# @weighted_loss
# def smooth_l1_loss_ohkm(pred, target, beta=1.0, top_k = 8):
#     assert beta > 0
#     assert pred.size() == target.size() and target.numel() > 0
#     diff = torch.abs(pred - target)
#     loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
#                        diff - 0.5 * beta)
#
#     # 待完善
#
#     # loss shape: 17 x 2
#     # OHKM
#
#     # loss_ohkm = ohkm(loss, top_k)
#
#     return loss


@LOSSES.register_module
class SmoothL1Loss(nn.Module):

    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * smooth_l1_loss(
            pred,
            target,
            weight,
            beta=self.beta,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_bbox
