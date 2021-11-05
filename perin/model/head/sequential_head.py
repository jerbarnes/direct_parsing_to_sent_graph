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

    # def anchor_cost_matrix(self, output, batch, decoder_lens, b: int):
    #     num_nodes = batch["labels"][1][b]
    #     word_lens = batch["every_input"][1][b]

    #     cost_matrix = 0.0
    #     for anchor_mode in ("anchor", "source_anchor", "target_anchor"):
    #         tgt_align = batch[anchor_mode][0][b, :num_nodes, :word_lens]  # shape: (num_nodes, num_inputs)
    #         align_prob = output[anchor_mode][b, :decoder_lens[b], :word_lens].sigmoid()  # shape: (num_queries, num_inputs)
    #         align_prob = align_prob.unsqueeze(1).expand(-1, num_nodes, -1)  # shape: (num_queries, num_nodes, num_inputs)
    #         align_prob = torch.where(tgt_align.unsqueeze(0).bool(), align_prob, 1.0 - align_prob)  # shape: (num_queries, num_nodes, num_inputs)
    #         cost_matrix += align_prob.log().mean(-1)  # shape: (num_queries, num_nodes)

    #     cost_matrix = (cost_matrix / 3).exp()
    #     return cost_matrix
