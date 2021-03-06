#!/usr/bin/env python3
# coding=utf-8

import torch
import torch.nn.functional as F


def masked_sum(loss, mask, label_weight=1, eps=1e-8, reduction=True):
    if mask is not None:
        loss = loss.masked_fill(mask, 0.0)
        if reduction:
            return loss.sum() / (((1 - mask.long()) * label_weight).sum() + eps)

    if reduction:
        return loss.mean()

    return loss


def cross_entropy(log_prob, target, mask, focal=False, label_weight=None, reduction=True):
    target = target.unsqueeze(-1)
    if focal:
        focal_coeff = log_prob.exp().gather(-1, target).squeeze(-1)
        focal_coeff = (1.0 - focal_coeff) ** 2
    else:
        focal_coeff = 1.0

    loss = -focal_coeff * log_prob.gather(-1, target).squeeze(-1)

    if label_weight is not None:
        loss = loss * label_weight
        return masked_sum(loss, mask, label_weight=label_weight, reduction=reduction)
    else:
        return masked_sum(loss, mask, reduction=reduction)


def binary_cross_entropy(logits, target, mask, focal=False, reduction=True):
    if focal:
        prob = logits.sigmoid()
        focal_coeff = target * prob + (1.0 - target) * (1.0 - prob)
        focal_coeff = (1.0 - focal_coeff) ** 2
    else:
        focal_coeff = 1.0

    loss = focal_coeff * F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return masked_sum(loss, mask, reduction=reduction)
