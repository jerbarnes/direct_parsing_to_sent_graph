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

from model.head.abstract_head import AbstractHead
from model.module.grad_scaler import scale_grad
from data.parser.to_mrp.sequential_parser import SequentialParser
from utility.cross_entropy import binary_cross_entropy


class SequentialHead(AbstractHead):
    def __init__(self, dataset, args, initialize):
        config = {
            "label": True,
            "property": False,
            "edge presence": False,
            "edge label": False,
            "anchor": True,
            "source_anchor": True,
            "target_anchor": True
        }
        super(SequentialHead, self).__init__(dataset, args, config, initialize)
        self.parser = SequentialParser(dataset,)

    def forward_label(self, decoder_output, decoder_lens):
        decoder_output = scale_grad(decoder_output, self.loss_weights["label"])
        return torch.log_softmax(self.label_classifier(decoder_output), dim=-1)

    def inference_label(self, prediction):
        prediction = prediction.exp()
        return torch.where(
            prediction[:, 0] > prediction[:, 1:].sum(-1),
            torch.zeros(prediction.size(0), dtype=torch.long, device=prediction.device),
            prediction[:, 1:].argmax(dim=-1) + 1
        )
