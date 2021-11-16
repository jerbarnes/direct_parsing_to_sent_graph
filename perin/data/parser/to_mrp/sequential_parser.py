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


class SequentialParser(AbstractParser):
    def parse(self, prediction):
        output = {}

        output["id"] = self.dataset.id_field.vocab.itos[prediction["id"].item()]
        output["nodes"] = self.create_nodes(prediction)
        output["nodes"] = self.create_anchors(prediction, output["nodes"], join_contiguous=True, at_least_one=True, mode="anchors")
        output["nodes"] = self.create_anchors(prediction, output["nodes"], join_contiguous=True, at_least_one=False, mode="source anchors")
        output["nodes"] = self.create_anchors(prediction, output["nodes"], join_contiguous=True, at_least_one=False, mode="target anchors")
        output["nodes"] = self.create_properties(prediction, output["nodes"])
        output["edges"], output["nodes"] = self.create_targets_sources(output["nodes"])

        return output

    def create_targets_sources(self, nodes):
        edges, new_nodes = [], []
        for i, node in enumerate(nodes):
            new_node_id = len(nodes) + len(new_nodes)
            if len(node["source anchors"]) > 0:
                new_nodes.append({"id": new_node_id, "label": "Source", "anchors": node["source anchors"]})
                edges.append({"source": i, "target": new_node_id, "label": ""})
                new_node_id += 1
            del node["source anchors"]

            if len(node["target anchors"]) > 0:
                new_nodes.append({"id": new_node_id, "label": "Target", "anchors": node["target anchors"]})
                edges.append({"source": i, "target": new_node_id, "label": ""})
            del node["target anchors"]

        return edges, nodes + new_nodes

    def create_properties(self, prediction, nodes):
        if prediction["properties"] is None:
            return nodes

        for node in nodes:
            node_properties = {}
            for key in prediction["properties"].keys():
                prop = self.dataset.property_field.vocabs[key].itos[prediction["properties"][key][node["id"], :].argmax()]
                if prop == "<NONE>":
                    continue
                node_properties[key] = prop

            if len(node_properties) > 0:
                node["properties"] = list(node_properties.keys())
                node["values"] = list(node_properties.values())

        return nodes
