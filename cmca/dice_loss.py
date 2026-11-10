"""Multiclass Dice loss; ignored pixels never mutate the input target."""
import torch
import torch.nn.functional as F


def dice_loss(output, target, eps=1e-7):
    valid = target != 255
    safe_target = target.masked_fill(~valid, 0).long()
    one_hot = F.one_hot(safe_target, output.shape[1]).permute(0, 3, 1, 2).to(output.dtype)
    mask = valid[:, None].to(output.dtype)
    prediction = output.softmax(1) * mask
    one_hot = one_hot * mask
    intersection = (prediction * one_hot).sum(dim=(0, 2, 3))
    denominator = (prediction + one_hot).sum(dim=(0, 2, 3))
    if not valid.any():
        return output.sum() * 0
    return 1 - (2 * intersection / (denominator + eps)).mean()


class DiceLoss(torch.nn.Module):
    def forward(self, output, target):
        return dice_loss(output, target)
