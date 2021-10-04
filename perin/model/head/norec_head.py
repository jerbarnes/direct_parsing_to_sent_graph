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
from data.parser.to_mrp.norec_parser import NorecParser


class NorecHead(AbstractHead):
    def __init__(self, dataset, args, framework, language, initialize):
        config = {
            "label": True,
            "property": False,
            "edge presence": True,
            "edge label": False,
            "anchor": True
        }
        super(NorecHead, self).__init__(dataset, args, framework, language, config, initialize)

        self.parser = NorecParser(dataset, language)

    def forward_label(self, decoder_output, decoder_lens):
        if self.label_classifier is None:
            return None
        decoder_output = scale_grad(decoder_output, self.loss_weights["label"])
        return torch.log_softmax(self.label_classifier(decoder_output), dim=-1)

    def inference_label(self, prediction):
        return prediction.argmax(dim=-1)
