#!/usr/bin/env python3
# conding=utf-8
#
# Copyright 2020 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from data.parser.to_mrp.abstract_parser import AbstractParser


class NodeCentricParser(AbstractParser):
    def parse(self, prediction):
        output = {}

        output["id"] = self.dataset.id_field.vocab.itos[prediction["id"].item()]
        output["nodes"] = self.create_nodes(prediction)
        output["nodes"] = self.create_anchors(prediction, output["nodes"], join_contiguous=True, at_least_one=True)
        output["edges"] = self.create_edges(prediction, output["nodes"])

        return output

    def create_edge(self, source, target, prediction, edges, nodes):
        edge = {"source": source, "target": target, "label": None}
        edges.append(edge)

    def create_edges(self, prediction, nodes):
        N = len(nodes)
        edge_prediction = prediction["edge presence"][:N, :N]

        targets = [i for i, node in enumerate(nodes) if node["label"] in ["Source", "Target"]]
        sources = [i for i, node in enumerate(nodes) if node["label"] not in ["Source", "Target"]]

        edges = []
        for target in targets:
            for source in sources:
                if edge_prediction[source, target] >= 0.5:
                    self.create_edge(source, target, prediction, edges, nodes)

        return edges
