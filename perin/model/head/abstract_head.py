#!/usr/bin/env python3
# conding=utf-8
#
# Copyright 2020 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.module.edge_classifier import EdgeClassifier
from model.module.anchor_classifier import AnchorClassifier
from model.module.padding_packer import PaddingPacker
from model.module.grad_scaler import scale_grad
from model.module.cross_entropy import multi_label_cross_entropy, cross_entropy, binary_cross_entropy
from utility.hungarian_matching import get_matching, reorder, match_anchor, match_label
from utility.utils import create_padding_mask


class AbstractHead(nn.Module):
    def __init__(self, dataset, args, framework, language, config, initialize: bool):
        super(AbstractHead, self).__init__()

        self.loss_weights = self.init_loss_weights(config)
        self.preference_weights = {}
        self.loss_0 = {}

        self.edge_classifier = self.init_edge_classifier(dataset, args, config, initialize)
        self.label_classifier = self.init_label_classifier(dataset, args, config, initialize)
        self.property_classifier = self.init_property_classifier(dataset, args, config, initialize)
        self.anchor_classifier = self.init_anchor_classifier(dataset, args, config, initialize)

        print(self.loss_0, flush=True)
        s = sum(self.preference_weights.values())
        for k in self.preference_weights.keys():
            self.preference_weights[k] /= s
        print(self.preference_weights, flush=True)

        self.query_length = args.query_length
        self.label_smoothing = args.label_smoothing
        self.focal = args.focal
        self.blank_weight = args.blank_weight
        self.dataset = dataset
        self.framework = framework
        self.language = language

    def forward(self, encoder_output, decoder_output, encoder_mask, decoder_mask, batch):
        output = {}

        decoder_lens = self.query_length * batch["every_input"][1]
        output["label"] = self.forward_label(decoder_output, decoder_lens)
        output["anchor"] = self.forward_anchor(decoder_output, encoder_output, encoder_mask)  # shape: (B, T_l, T_w)

        cost_matrices = self.create_cost_matrices(output, batch, decoder_lens)
        matching = get_matching(cost_matrices)
        decoder_output = reorder(decoder_output, matching, batch["labels"][0].size(1))

        output["property"] = self.forward_property(decoder_output)
        output["edge presence"], output["edge label"] = self.forward_edge(decoder_output)

        #try:
        return self.loss(output, batch, matching, decoder_mask)
        # except:
        #     print(batch)
        #     print()
        #     print(output)
        #     print()
        #     print(decoder_mask)
        #     print()
        #     print(matching)
        #     print()
        #     print(decoder_output.shape)
        #     exit()

    def predict(self, encoder_output, decoder_output, encoder_mask, decoder_mask, batch, **kwargs):
        every_input, word_lens = batch["every_input"]
        decoder_lens = self.query_length * word_lens
        batch_size = every_input.size(0)

        label_pred = self.forward_label(decoder_output, decoder_lens)
        anchor_pred = self.forward_anchor(decoder_output, encoder_output, encoder_mask)  # shape: (B, T_l, T_w)

        labels, anchors = [[] for _ in range(batch_size)], [[] for _ in range(batch_size)]
        for b in range(batch_size):
            label_indices = self.inference_label(label_pred[b, :decoder_lens[b], :]).cpu()
            for t in range(label_indices.size(0)):
                label_index = label_indices[t].item()
                if label_index == 0:
                    continue

                decoder_output[b, len(labels[b]), :] = decoder_output[b, t, :]

                labels[b].append(label_index)
                if anchor_pred is None:
                    anchors[b].append(list(range(t // self.query_length, word_lens[b])))
                else:
                    anchors[b].append(self.inference_anchor(anchor_pred[b, t, :word_lens[b]]).cpu())

        decoder_output = decoder_output[:, : max(len(l) for l in labels), :]

        properties = self.forward_property(decoder_output)
        edge_presence, edge_labels = self.forward_edge(decoder_output)

        outputs = [
            self.parser.parse(
                {
                    "labels": labels[b],
                    "anchors": anchors[b],
                    "properties": self.inference_property(properties, b),
                    "edge presence": self.inference_edge_presence(edge_presence, b),
                    "edge labels": self.inference_edge_label(edge_labels, b),
                    "id": batch["id"][b].cpu(),
                    "tokens": batch["every_input"][0][b, : word_lens[b]].cpu(),
                    "token intervals": batch["token_intervals"][b, :, :].cpu(),
                },
                **kwargs
            )
            for b in range(batch_size)
        ]

        return outputs

    def loss(self, output, batch, matching, decoder_mask):
        batch_size = batch["every_input"][0].size(0)
        device = batch["every_input"][0].device
        T_label = batch["labels"][0].size(1)
        T_input = batch["every_input"][0].size(1)

        input_mask = create_padding_mask(batch_size, T_input, batch["every_input"][1], device)  # shape: (B, T_input)
        label_mask = create_padding_mask(batch_size, T_label, batch["labels"][1], device)  # shape: (B, T_label)
        edge_mask = torch.eye(T_label, T_label, device=device, dtype=torch.bool).unsqueeze(0)  # shape: (1, T_label, T_label)
        edge_mask = edge_mask | label_mask.unsqueeze(1) | label_mask.unsqueeze(2)  # shape: (B, T_label, T_label)
        edge_label_mask = (batch["edge_presence"] == 0) | edge_mask

        if output["edge label"] is not None:
            batch["edge_labels"] = (
                batch["edge_labels"][0][:, :, :, :output["edge label"].size(-1)],
                batch["edge_labels"][1],
            )

        losses = {}
        losses.update(self.loss_label(output, batch, decoder_mask, matching))
        losses.update(self.loss_anchor(output, batch, input_mask, matching))
        losses.update(self.loss_edge_presence(output, batch, edge_mask))
        losses.update(self.loss_edge_label(output, batch, edge_label_mask.unsqueeze(-1)))
        losses.update(self.loss_property(output, batch, label_mask))

        stats = {f"{key}": value.detach().cpu().item() for key, value in losses.items()}
        total_loss = sum(losses.values())

        return total_loss, losses, stats

    @torch.no_grad()
    def create_cost_matrices(self, output, batch, decoder_lens):
        batch_size = len(batch["labels"][1])
        decoder_lens = decoder_lens.cpu()

        matrices = []
        for b in range(batch_size):
            label_cost_matrix = self.label_cost_matrix(output, batch, decoder_lens, b)
            anchor_cost_matrix = self.anchor_cost_matrix(output, batch, decoder_lens, b)

            cost_matrix = label_cost_matrix * anchor_cost_matrix
            matrices.append(cost_matrix.cpu())

        return matrices

    def init_loss_weights(self, config):
        default_weight = 1.0 / len([v for v in config.values() if v])
        return nn.ParameterDict({k: nn.Parameter(torch.tensor([default_weight])) for k, v in config.items() if v})

    def init_edge_classifier(self, dataset, args, config, initialize: bool):
        if not config["edge presence"] and not config["edge label"]:
            return None
        if config["edge presence"]:
            self.preference_weights["edge presence"] = 1.0  # dataset.edge_count / math.sqrt(2)
            self.loss_0["edge presence"] = torch.distributions.bernoulli.Bernoulli(dataset.edge_presence_freq).entropy()
        if config["edge label"]:
            self.preference_weights["edge label"] = 1.0  # dataset.edge_count / math.sqrt(2)
            self.loss_0["edge label"] = torch.distributions.categorical.Categorical(dataset.edge_label_freqs).entropy() / 2

        return EdgeClassifier(dataset, args, initialize, presence=config["edge presence"], label=config["edge label"])

    def init_label_classifier(self, dataset, args, config, initialize: bool):
        if not config["label"]:
            return None

        self.preference_weights["label"] = 1.0  # dataset.node_count / math.sqrt(2)
        self.loss_0["label"] = torch.distributions.categorical.Categorical(dataset.label_freqs).entropy()  # / 2

        classifier = nn.Sequential(
            nn.Dropout(args.dropout_label),
            nn.Linear(args.hidden_size, len(dataset.label_field.vocab) + 1, bias=True)
        )
        if initialize:
            classifier[1].bias.data = dataset.label_freqs.log()

        return PaddingPacker(classifier)

    def init_property_classifier(self, dataset, args, config, initialize: bool):
        if not config["property"]:
            return None

        self.preference_weights["property"] = 1.0  # dataset.property_field.vocabs["transformed"].freqs[dataset.property_field.vocabs["transformed"].stoi[1]]

        classifier = nn.Sequential(nn.Dropout(args.dropout_property), nn.Linear(args.hidden_size, 1))

        if initialize:
            property_freq = dataset.property_freqs["transformed"][dataset.property_field.vocabs["transformed"].stoi[1]]
            classifier[1].bias.data.fill_((property_freq / (1.0 - property_freq)).log())
            self.loss_0["property"] = torch.distributions.bernoulli.Bernoulli(property_freq).entropy()

        return classifier

    def init_anchor_classifier(self, dataset, args, config, initialize: bool):
        if not config["anchor"]:
            return None

        self.preference_weights["anchor"] = 1.0  # dataset.node_count / math.sqrt(2)
        self.loss_0["anchor"] = torch.distributions.bernoulli.Bernoulli(dataset.anchor_freq).entropy()

        return AnchorClassifier(dataset, args, initialize)

    def forward_edge(self, decoder_output):
        if self.edge_classifier is None:
            return None
        return self.edge_classifier(decoder_output, self.loss_weights)

    def forward_label(self, decoder_output, decoder_lens):
        if self.label_classifier is None:
            return None
        decoder_output = scale_grad(decoder_output, self.loss_weights["label"])
        return self.label_classifier(decoder_output, decoder_lens, decoder_output.size(1))

    def forward_property(self, decoder_output):
        if self.property_classifier is None:
            return None
        decoder_output = scale_grad(decoder_output, self.loss_weights["property"])
        return self.property_classifier(decoder_output).squeeze(-1)

    def forward_anchor(self, decoder_output, encoder_output, encoder_mask):
        if self.anchor_classifier is None:
            return None
        decoder_output = scale_grad(decoder_output, self.loss_weights["anchor"])
        return self.anchor_classifier(decoder_output, encoder_output, encoder_mask)

    def inference_label(self, prediction):
        min_diff = (prediction[:, 0] - prediction[:, 1:].max(-1)[0]).min()
        if min_diff >= 0:
            prediction[:, 0] -= min_diff + 1e-3  # make sure at least one item will be selected

        return prediction.argmax(dim=-1)

    def inference_anchor(self, prediction):
        return prediction.sigmoid()

    def inference_property(self, prediction, example_index: int):
        if prediction is None:
            return None
        return prediction[example_index, :].sigmoid().cpu()

    def inference_edge_presence(self, prediction, example_index: int):
        if prediction is None:
            return None

        N = prediction.size(1)
        mask = torch.eye(N, N, device=prediction.device, dtype=torch.bool)
        return prediction[example_index, :, :].sigmoid().masked_fill(mask, 0.0).cpu()

    def inference_edge_label(self, prediction, example_index: int):
        if prediction is None:
            return None
        return prediction[example_index, :, :, :].argmax(dim=-1).cpu()

    def loss_edge_presence(self, prediction, target, mask):
        if self.edge_classifier is None or prediction["edge presence"] is None:
            return {}
        return {"edge presence": binary_cross_entropy(prediction["edge presence"], target["edge_presence"].float(), mask)}

    def loss_edge_label(self, prediction, target, mask):
        if self.edge_classifier is None or prediction["edge label"] is None:
            return {}
        return {"edge label": binary_cross_entropy(prediction["edge label"], target["edge_labels"][0].float(), mask)}

    def loss_label(self, prediction, target, mask, matching):
        if self.label_classifier is None or prediction["label"] is None:
            return {}

        prediction = prediction["label"]
        target, label_weight = match_label(
            target["labels"][0], matching, prediction.shape[:-1], prediction.device, self.query_length, self.blank_weight
        )
        return {"label": cross_entropy(prediction, target, mask, focal=self.focal, label_weight=label_weight)}

    def loss_property(self, prediction, target, mask):
        if self.property_classifier is None or prediction["property"] is None:
            return {}
        return {"property": binary_cross_entropy(prediction["property"], target["properties"][:, :, 0].float(), mask)}

    def loss_anchor(self, prediction, target, mask, matching):
        if self.anchor_classifier is None or prediction["anchor"] is None:
            return {}

        prediction = prediction["anchor"]
        target, anchor_mask = match_anchor(target["anchor"], matching, prediction.shape, prediction.device)
        mask = anchor_mask.unsqueeze(-1) | mask.unsqueeze(-2)
        return {"anchor": binary_cross_entropy(prediction, target.float(), mask)}

    def label_cost_matrix(self, output, batch, decoder_lens, b: int):
        if output["label"] is None:
            return 1.0

        indices = batch["labels"][0][b, :batch["labels"][1][b]]  # shape: (num_nodes)
        label_prob = output["label"][b, : decoder_lens[b], :]  # shape: (num_queries, num_classes)
        indices = indices.view(1, -1, 1).expand(label_prob.size(0), -1, -1)  # shape: (num_queries, num_nodes, 1)
        label_prob = label_prob.unsqueeze(1).expand(-1, indices.size(1), -1)  # shape: (num_queries, num_nodes, num_classes)
        cost_matrix = torch.gather(label_prob, 2, indices).squeeze(2).exp()  # shape: (num_queries, num_nodes)

        return cost_matrix

    def anchor_cost_matrix(self, output, batch, decoder_lens, b: int):
        if output["anchor"] is None:
            return 1.0

        num_nodes = batch["labels"][1][b]
        word_lens = batch["every_input"][1]
        target_anchors, _ = batch["anchor"]
        pred_anchors = output["anchor"].sigmoid()

        tgt_align = target_anchors[b, : num_nodes, : word_lens[b]]  # shape: (num_nodes, num_inputs)
        align_prob = pred_anchors[b, : decoder_lens[b], : word_lens[b]]  # shape: (num_queries, num_inputs)
        align_prob = align_prob.unsqueeze(1).expand(-1, num_nodes, -1)  # shape: (num_queries, num_nodes, num_inputs)
        align_prob = torch.where(tgt_align.unsqueeze(0).bool(), align_prob, 1.0 - align_prob)  # shape: (num_queries, num_nodes, num_inputs)
        cost_matrix = align_prob.log().mean(-1).exp()  # shape: (num_queries, num_nodes)
        return cost_matrix

    def loss_weights_dict(self):
        loss_weights = {f"weight/{key}": weight.detach().cpu().item() for key, weight in self.loss_weights.items()}
        return loss_weights
