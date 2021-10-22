#!/usr/bin/env python3
# conding=utf-8
#
# Copyright 2020 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

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


def multi_label_cross_entropy(probs, target, mask, focal=False, label_weight=None, reduction=True):
    if focal:
        focal_coeff = (probs * target).sum(dim=-1)
        focal_coeff = (1.0 - focal_coeff) ** 2
    else:
        focal_coeff = 1.0

    logits = torch.log(probs + 1e-8)
    loss = -focal_coeff * (logits * target).sum(dim=-1)

    if label_weight is not None:
        loss = loss * label_weight
        return masked_sum(loss, mask, label_weight=label_weight, reduction=reduction)
    else:
        return masked_sum(loss, mask, reduction=reduction)


def cross_entropy(log_prob, target, mask, focal=False, label_weight=None, reduction=True, smoothing=0.0):
    target = target.unsqueeze(-1)
    if focal:
        focal_coeff = log_prob.exp().gather(-1, target).squeeze(-1)
        focal_coeff = (1.0 - focal_coeff) ** 2
    else:
        focal_coeff = 1.0

    if smoothing == 0.0:
        loss = -focal_coeff * log_prob.gather(-1, target).squeeze(-1)
    else:
        loss = -focal_coeff * smooth_cross_entropy(log_prob, target, smoothing=smoothing)

    if label_weight is not None:
        loss = loss * label_weight
        return masked_sum(loss, mask, label_weight=label_weight, reduction=reduction)
    else:
        return masked_sum(loss, mask, reduction=reduction)


def smooth_cross_entropy(log_prob, gold, smoothing=0.1):
    n_class = log_prob.size(-1)

    one_hot = torch.full_like(log_prob, fill_value=smoothing / (n_class - 1))
    one_hot.scatter_(dim=-1, index=gold, value=1.0 - smoothing)

    return F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1)


def binary_cross_entropy(logits, target, mask, focal=False, reduction=True):
    if focal:
        prob = logits.sigmoid()
        focal_coeff = target * prob + (1.0 - target) * (1.0 - prob)
        focal_coeff = (1.0 - focal_coeff) ** 2
    else:
        focal_coeff = 1.0

    loss = focal_coeff * F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return masked_sum(loss, mask, reduction=reduction)
