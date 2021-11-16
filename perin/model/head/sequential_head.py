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
import torch.nn as nn
import torch.nn.functional as F

from model.head.abstract_head import AbstractHead
from model.module.grad_scaler import scale_grad
from data.parser.to_mrp.sequential_parser import SequentialParser
from utility.cross_entropy import binary_cross_entropy, cross_entropy


class SequentialHead(AbstractHead):
    def __init__(self, dataset, args, initialize):
        config = {
            "label": True,
            "property": args.predict_intensity,
            "edge presence": False,
            "edge label": False,
            "anchor": True,
            "source_anchor": True,
            "target_anchor": True
        }
        super(SequentialHead, self).__init__(dataset, args, config, initialize)
        self.property_key = dataset.property_keys[0]
        self.parser = SequentialParser(dataset)

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

    def init_property_classifier(self, dataset, args, config, initialize: bool):
        if not config["property"]:
            return None

        classifiers = nn.ModuleDict({
            key: nn.Sequential(nn.Dropout(args.dropout_property), nn.Linear(args.hidden_size, len(vocab)))
            for key, vocab in dataset.property_field.vocabs.items()
        })

        for key, vocab in dataset.property_field.vocabs.items():
            self.preference_weights[f"property {key}"] = dataset.node_count - vocab.freqs["<NONE>"]

        if initialize:
            for key, vocab in dataset.property_field.vocabs.items():
                self.preference_weights[f"property {key}"] = dataset.node_count - vocab.freqs["<NONE>"]
                self.loss_0[f"property {key}"] = torch.distributions.categorical.Categorical(dataset.property_freqs[key]).entropy()

            for key, freq in dataset.property_freqs.items():
                classifiers[key][1].bias.data = freq.log()
                classifiers[key][1].weight.data *= 0.01

        return classifiers

    def forward_property(self, decoder_output):
        if self.property_classifier is None:
            return None
        output = {}
        # scaled_decoder_output = scale_grad(decoder_output, self.loss_weights[f"property {key}"])
        output[f"{self.property_key}"] = F.log_softmax(self.property_classifier[self.property_key](decoder_output), dim=-1)

        return output

    def inference_property(self, prediction, example_index: int):
        if prediction is None:
            return None
        return {self.property_key: F.softmax(prediction[self.property_key][example_index, :, :], dim=-1).cpu()}

    def loss_property(self, prediction, target, mask):
        if self.property_classifier is None or prediction["property"] is None:
            return {}
        loss = {}
        loss[f"property {self.property_key}"] = cross_entropy(prediction["property"][self.property_key], target["properties"].squeeze(-1), mask)
        return loss
