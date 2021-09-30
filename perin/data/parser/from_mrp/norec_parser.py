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

        utils.add_companion(self.data, None, self.language, tokenization_mode="space")  # add empty companion
        utils.tokenize(self.data, mode="space")

        utils.anchor_ids_from_intervals(self.data)

        self.node_counter, self.edge_counter, self.no_edge_counter = 0, 0, 0
        anchor_count, n_node_token_pairs = 0, 0

        for node, sentence in utils.node_generator(self.data):
            self.node_counter += 1

        utils.create_bert_tokens(self.data, args.encoder)
        utils.create_edge_permutations(self.data, NorecParser.node_similarity_key)

        # create edge vectors
        for sentence in self.data.values():
            N = len(sentence["nodes"])

            edge_count = utils.create_edges(sentence, attributes=False, normalize=False)
            self.edge_counter += edge_count
            self.no_edge_counter += N * (N - 1) - edge_count

            sentence["anchor edges"] = [N, len(sentence["input"]), []]
            for i, node in enumerate(sentence["nodes"]):
                for anchor in node["anchors"]:
                    sentence["anchor edges"][-1].append((i, anchor))

                anchor_count += len(node["anchors"])
                n_node_token_pairs += len(sentence["input"])

            sentence["id"] = [sentence["id"]]
            sentence["top"] = 0  # we don't need it for Norec

        self.anchor_freq = anchor_count / n_node_token_pairs
        self.input_count = sum(len(sentence["input"]) for sentence in self.data.values())

        super(NorecParser, self).__init__(fields, self.data, filter_pred)

    @staticmethod
    def node_similarity_key(node):
        return tuple([node["label"]] + node["anchors"])
