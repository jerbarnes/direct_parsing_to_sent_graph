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
from data.parser.to_mrp.node_centric_parser import NodeCentricParser
from utility.cross_entropy import binary_cross_entropy


class NodeCentricHead(AbstractHead):
    def __init__(self, dataset, args, initialize):
        config = {
            "label": True,
            "property": False,
            "edge presence": True,
            "edge label": False,
            "anchor": True,
            "source_anchor": False,
            "target_anchor": False
        }
        super(NodeCentricHead, self).__init__(dataset, args, config, initialize)

        self.source_id = dataset.label_field.vocab.stoi["Source"] + 1
        self.target_id = dataset.label_field.vocab.stoi["Target"] + 1
        self.parser = NodeCentricParser(dataset)

    def forward_label(self, decoder_output, decoder_lens):
        decoder_output = scale_grad(decoder_output, self.loss_weights["label"])
        return torch.log_softmax(self.label_classifier(decoder_output), dim=-1)

    def loss_edge_presence(self, prediction, target, mask):
        # source_target_mask = (target["labels"][0] == self.source_id) | (target["labels"][0] == self.target_id)  # shape: (B, T_label)
        # mask = mask | source_target_mask.unsqueeze(2)
        # mask = mask | (~source_target_mask).unsqueeze(1)
        return {"edge presence": binary_cross_entropy(prediction["edge presence"], target["edge_presence"].float(), mask)}

    def inference_label(self, prediction):
        prediction = prediction.exp()
        return torch.where(
            prediction[:, 0] > prediction[:, 1:].sum(-1),
            torch.zeros(prediction.size(0), dtype=torch.long, device=prediction.device),
            prediction[:, 1:].argmax(dim=-1) + 1
        )

    # def inference_label(self, prediction):
    #     return prediction.argmax(dim=-1)
