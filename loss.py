import torch.nn as nn
import torch
import math


class SoftLoULoss1(nn.Module):
    def __init__(self, batch=32):
        super(SoftLoULoss1, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()

    def forward(self, pred, target):
        # pred, target 为概率
        smooth = 0.0
        intersection = pred * target
        intersection_sum = torch.sum(intersection, dim=(1,2,3))
        pred_sum = torch.sum(pred, dim=(1,2,3))
        target_sum = torch.sum(target, dim=(1,2,3))
        iou = (intersection_sum + smooth) / (pred_sum + target_sum - intersection_sum + smooth)
        loss = 1 - torch.mean(iou)
        return loss
class SLSIoULoss(nn.Module):
    def __init__(self):
        super(SLSIoULoss, self).__init__()

    def forward(self, pred_prob, target, warm_epoch=40, epoch=1, with_distance=True, dynamic=True, delta=0.5):
        # pred_prob 已经是概率 ∈ [0,1]
        pred = pred_prob.clamp(1e-6, 1 - 1e-6)
        target = target.clamp(0, 1)

        intersection = (pred * target).sum(dim=[1, 2, 3])
        pred_sum = pred.sum(dim=[1, 2, 3])
        target_sum = target.sum(dim=[1, 2, 3])
        union = pred_sum + target_sum - intersection + 1e-6

        siou = intersection / union
        siou_loss = 1 - siou.mean()

        if epoch > warm_epoch and dynamic and with_distance:
            lloss = LLoss(pred, target)
            # beta 根据目标面积适配
            beta = (target_sum.mean() * delta * (512 * 512) / (pred.shape[2] * pred.shape[3])) / 81
            beta = torch.clamp(beta, max=delta)
            loss = (1 + beta) * siou_loss + (1 - beta) * lloss
        else:
            loss = siou_loss

        return loss


def LLoss(pred, target):
    eps = 1e-6
    device = pred.device
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)

    B, H, W = pred.shape
    x_index = torch.linspace(0, 1, W, device=device).view(1, 1, W).expand(B, H, W)
    y_index = torch.linspace(0, 1, H, device=device).view(1, H, 1).expand(B, H, W)

    total_loss = pred.new_tensor(0.0)
    count = 0

    for i in range(B):
        p_sum = pred[i].sum()
        t_sum = target[i].sum()
        if p_sum < eps or t_sum < eps:
            continue

        px = (x_index[i] * pred[i]).sum() / p_sum
        py = (y_index[i] * pred[i]).sum() / p_sum
        tx = (x_index[i] * target[i]).sum() / t_sum
        ty = (y_index[i] * target[i]).sum() / t_sum

        pred_angle = torch.atan2(py.clamp(-10, 10), px.clamp(-10, 10))
        true_angle = torch.atan2(ty.clamp(-10, 10), tx.clamp(-10, 10))
        angle_diff = pred_angle - true_angle
        angle_loss = (4.0 / (math.pi ** 2)) * angle_diff ** 2

        pred_len = torch.sqrt(px ** 2 + py ** 2 + eps)
        true_len = torch.sqrt(tx ** 2 + ty ** 2 + eps)
        length_ratio = torch.min(pred_len, true_len) / (torch.max(pred_len, true_len) + eps)
        length_loss = 1.0 - length_ratio

        total_loss = total_loss + (length_loss + angle_loss)
        count += 1

    if count > 0:
        return total_loss / count
    else:
        # 没有目标时，返回 0，但保持梯度通路
        return pred.sum() * 0.0


class SGFRNetLoss(nn.Module):
    """Loss used by the released SGFRNet training entry point.

    The main output and four deep-supervision outputs are aggregated with the
    same descending weights used in the original experiment script.
    """

    def __init__(self, warm_epoch=40, deep_supervision_weights=None):
        super().__init__()
        self.warm_epoch = warm_epoch
        self.weights = deep_supervision_weights or [1.0, 0.5, 0.25, 0.125, 0.0625]
        self.bce = nn.BCELoss()
        self.sls_iou = SLSIoULoss()

    @staticmethod
    def dice_loss(pred, target, smooth=1e-5):
        pred = pred.contiguous().view(pred.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)
        intersection = (pred * target).sum(dim=1)
        dice = (2.0 * intersection + smooth) / (
            pred.sum(dim=1) + target.sum(dim=1) + smooth
        )
        return 1.0 - dice.mean()

    def forward(self, predictions, target, epoch):
        outputs = list(predictions) if isinstance(predictions, (tuple, list)) else [predictions]
        if len(outputs) > len(self.weights):
            raise ValueError("More deep-supervision outputs than configured weights")

        active_weights = self.weights[:len(outputs)]
        normalizer = sum(active_weights)
        bce_dice = target.new_tensor(0.0)
        sls_iou = target.new_tensor(0.0)

        for weight, prediction in zip(active_weights, outputs):
            bce_dice = bce_dice + weight * (
                self.bce(prediction, target) + self.dice_loss(prediction, target)
            )
            sls_iou = sls_iou + weight * self.sls_iou(
                prediction,
                target,
                warm_epoch=self.warm_epoch,
                epoch=epoch,
                delta=0.5,
            )

        bce_dice = bce_dice / normalizer
        sls_iou = sls_iou / normalizer
        sls_weight = 0.5 if epoch <= self.warm_epoch else 1.0
        total = bce_dice + sls_weight * sls_iou
        return {"total": total, "bce_dice": bce_dice, "sls_iou": sls_iou}

