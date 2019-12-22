import torch.nn as nn
import torch.nn.functional as F
import torch
import copy

from ..registry import LOSSES
from .utils import weighted_loss


def ohkm(loss, top_k):
    # ohkm_loss = copy.deepcopy(loss)
    loss_point = loss.mean((1), keepdim=True)
    topk_val, topk_idx = torch.topk(loss_point, k=top_k, dim=0, sorted=False)

    tmp_loss = torch.gather(loss_point, 0, topk_idx)
    ohkm_loss = torch.sum(tmp_loss) / top_k

    return ohkm_loss/10.0

def mse_loss_ohkm(pred, target, weights, top_k=17):
    assert pred.size() == target.size() and target.numel() > 0
    # diff = torch.abs(pred - target)

    loss = F.mse_loss(pred, target, reduction='none')
    loss_weight = loss*weights

    loss_ohkm = ohkm(loss_weight, top_k)

    return loss_ohkm

# mse_loss_ohkm = weighted_loss(mse_loss_ohkm)
mse_loss = weighted_loss(F.mse_loss)


@LOSSES.register_module
class MSELoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None):
        loss = self.loss_weight * mse_loss(
            pred,
            target,
            weight,
            reduction=self.reduction,
            avg_factor=avg_factor)
        return loss
