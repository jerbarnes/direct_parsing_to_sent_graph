#!/usr/bin/env python3
# conding=utf-8
#
# Copyright 2020 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

class AbstractParser:
    def __init__(self, dataset):
        self.dataset = dataset

    def create_nodes(self, prediction):
        return [
            {"id": i, "label": self.label_to_str(l, prediction["anchors"][i], prediction)}
            for i, l in enumerate(prediction["labels"])
        ]

    def label_to_str(self, label, anchors, prediction):
        return self.dataset.label_field.vocab.itos[label - 1]

    def create_properties(self, prediction, nodes):
        N = len(nodes)
        properties_p = prediction["properties"][:N]

        threshold = max(0.5, properties_p.min().item())  # make sure there's at least one proper node
        properties = (properties_p > threshold).nonzero(as_tuple=False).squeeze(-1)
        non_properties = (properties_p <= threshold).nonzero(as_tuple=False).squeeze(-1)

        for i in properties:
            edge_probs = prediction["edge presence"][:N, i].clone()
            edge_probs[properties] = 0.0

            parent_i = edge_probs.argmax()
            parent = nodes[parent_i]
            if "properties" not in parent:
                parent["properties"], parent["values"] = [], []
            parent["properties"].append(self.get_edge_label(prediction, parent_i, i))
            parent["values"].append(nodes[i]["label"])

        nodes = [nodes[i] for i in non_properties]
        for i in range(len(nodes)):
            nodes[i]["id"] = i

        prediction["edge presence"] = prediction["edge presence"][non_properties, :][:, non_properties]
        prediction["edge labels"] = prediction["edge labels"][non_properties, :][:, non_properties]
        if "anchors" in prediction and prediction["anchors"] is not None:
            prediction["anchors"] = [prediction["anchors"][i] for i in non_properties]

        return nodes

    def create_edges(self, prediction, nodes):
        N = len(nodes)
        node_sets = [{"id": n, "set": set([n])} for n in range(N)]
        _, indices = prediction["edge presence"][:N, :N].reshape(-1).sort(descending=True)
        sources, targets = indices // N, indices % N

        edges = []
        for i in range((N - 1) * N // 2):
            source, target = sources[i].item(), targets[i].item()
            p = prediction["edge presence"][source, target]

            if p < 0.5 and len(edges) >= N - 1:
                break

            if node_sets[source]["set"] is node_sets[target]["set"] and p < 0.5:
                continue

            self.create_edge(source, target, prediction, edges, nodes)

            if node_sets[source]["set"] is not node_sets[target]["set"]:
                from_set = node_sets[source]["set"]
                for n in node_sets[target]["set"]:
                    from_set.add(n)
                    node_sets[n]["set"] = from_set

        return edges

    def create_edge(self, source, target, prediction, edges, nodes):
        label = self.get_edge_label(prediction, source, target)
        edge = {"source": source, "target": target, "label": label}

        edges.append(edge)

    def create_anchors(self, prediction, nodes, join_contiguous=True, at_least_one=False, single_anchor=False, mode="anchors"):
        for i, node in enumerate(nodes):
            threshold = 0.5 if not at_least_one else min(0.5, prediction[mode][i].max().item())
            node[mode] = (prediction[mode][i] >= threshold).nonzero(as_tuple=False).squeeze(-1)
            node[mode] = prediction["token intervals"][node[mode], :]

            if single_anchor and len(node[mode]) > 1:
                start = min(a[0].item() for a in node[mode])
                end = max(a[1].item() for a in node[mode])
                node[mode] = [{"from": start, "to": end}]
                continue

            node[mode] = [{"from": f.item(), "to": t.item()} for f, t in node[mode]]
            node[mode] = sorted(node[mode], key=lambda a: a["from"])

            if join_contiguous and len(node[mode]) > 1:
                cleaned_anchors = []
                end, start = node[mode][0]["from"], node[mode][0]["from"]
                for anchor in node[mode]:
                    if end < anchor["from"]:
                        cleaned_anchors.append({"from": start, "to": end})
                        start = anchor["from"]
                    end = anchor["to"]
                cleaned_anchors.append({"from": start, "to": end})

                node[mode] = cleaned_anchors

        return nodes

    def get_edge_label(self, prediction, source, target):
        return self.dataset.edge_label_field.vocab.itos[prediction["edge labels"][source, target].item()]
