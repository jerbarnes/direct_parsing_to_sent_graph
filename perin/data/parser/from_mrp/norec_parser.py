#!/usr/bin/env python3
# conding=utf-8
#
# Copyright 2020 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import json
import io
import os
import os.path

from collections import Counter
from data.parser.from_mrp.abstract_parser import AbstractParser
import utility.parser_utils as utils


class NorecParser(AbstractParser):
    def __init__(self, args, framework: str, language: str, part: str, fields, precomputed_dataset=None, filter_pred=None, **kwargs):
        assert part == "training" or part == "validation"
        path = args.training_data[(framework, language)] if part == "training" else args.validation_data[(framework, language)]

        self.framework = framework
        self.language = language

        self.data = utils.load_dataset(path, framework=self.framework)

        utils.add_fake_companion(self.data, self.language, tokenization_mode="space")  # add empty companion
        utils.tokenize(self.data, mode="space")

        utils.anchor_ids_from_intervals(self.data)

        self.node_counter, self.edge_counter, self.no_edge_counter = 0, 0, 0
        anchor_count, n_node_token_pairs = 0, 0

        unlabeled_count = 0
        for node, sentence in utils.node_generator(self.data):
            if "label" not in node:
                node["label"] = "Null"
                unlabeled_count += 1
            node["properties"] = {"dummy": 0}

            self.node_counter += 1
        # print(f"Number of unlabeled nodes: {unlabeled_count}", flush=True)

        utils.create_bert_tokens(self.data, args.encoder)

        # create edge vectors
        for sentence in self.data.values():
            sentence["language"] = language  # just to get through the later checks...
            N = len(sentence["nodes"])

            edge_count = utils.create_edges(sentence, normalize=False)
            self.edge_counter += edge_count
            # self.no_edge_counter += len([n for n in sentence["nodes"] if n["label"] in ["Source", "Target"]]) * len([n for n in sentence["nodes"] if n["label"] not in ["Source", "Target"]]) - edge_count
            self.no_edge_counter += N * (N - 1) - edge_count

            sentence["anchor edges"] = [N, len(sentence["input"]), []]
            sentence["anchored labels"] = [len(sentence["input"]), []]
            for i, node in enumerate(sentence["nodes"]):
                anchored_labels = []
                #if len(node["anchors"]) == 0:
                #    print(f"Empty node in {sentence['id']}", flush=True)

                for anchor in node["anchors"]:
                    sentence["anchor edges"][-1].append((i, anchor))
                    anchored_labels.append((anchor, node["label"]))

                sentence["anchored labels"][1].append(anchored_labels)

                anchor_count += len(node["anchors"])
                n_node_token_pairs += len(sentence["input"])

            sentence["id"] = [sentence["id"]]

        self.anchor_freq = anchor_count / n_node_token_pairs
        self.input_count = sum(len(sentence["input"]) for sentence in self.data.values())

        super(NorecParser, self).__init__(fields, self.data, filter_pred)

    @staticmethod
    def node_similarity_key(node):
        return tuple([node["label"]] + node["anchors"])
