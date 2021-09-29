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
from utility.label_processor import LabelProcessor


class NorecParser(AbstractParser):
    def __init__(self, args, framework: str, language: str, part: str, fields, precomputed_dataset=None, filter_pred=None, **kwargs):
        assert part == "training" or part == "validation"
        path = args.training_data[(framework, language)] if part == "training" else args.validation_data[(framework, language)]

        self.framework = framework
        self.language = language

        cache_path = f"{path}_cache"
        if not os.path.exists(cache_path):
            self.initialize(args, path, cache_path, args.companion_data[(framework, language)], precomputed_dataset=precomputed_dataset)

        print("Loading the cached dataset")

        self.data = {}
        with io.open(cache_path, encoding="utf8") as reader:
            for line in reader:
                sentence = json.loads(line)
                self.data[sentence["id"]] = sentence

        self.node_counter, self.edge_counter, self.no_edge_counter = 0, 0, 0
        anchor_count, n_node_token_pairs = 0, 0

        for node, sentence in utils.node_generator(self.data):
            self.node_counter += 1

        utils.create_bert_tokens(self.data, args.encoder)

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

    def initialize(self, args, raw_path, cache_path, companion_path, precomputed_dataset=None):
        print("Caching the dataset...\n", flush=True)

        data = utils.load_dataset(raw_path, framework=self.framework)

        utils.tokenize(data, mode="space")
        utils.add_companion(data, None, self.language)  # add empty companion
        utils.anchor_ids_from_intervals(data)

        with open(cache_path, "w", encoding="utf8") as f:
            for example in data.values():
                json.dump(example, f, ensure_ascii=False)
                f.write("\n")

    @staticmethod
    def _create_possible_rules(node, sentence):
        processor = LabelProcessor()

        anchors = node["anchors"]
        if len(anchors) == 0:
            return [{"rule": processor.make_absolute_label_rule(node["label"].lower()), "anchor": None}]

        rules = processor.gen_all_label_rules(
            [sentence["input"][anchor] for anchor in anchors],
            [sentence["lemmas"][anchor] for anchor in anchors],
            node["label"],
            separators=['', '+', '-'],
            rule_classes=["absolute", "relative_forms", "relative_lemmas", "numerical_all"],
            # separators=['', '-'],
            # rule_classes=["absolute", "relative_forms", "numerical_all"],
            concat=True,
            allow_copy=False,
            ignore_nonalnum=True,
        )
        return [{"rule": rule, "anchor": node["anchors"]} for rule in set(rules)]

    @staticmethod
    def node_similarity_key(node):
        return tuple([node["label"]] + node["anchors"])
