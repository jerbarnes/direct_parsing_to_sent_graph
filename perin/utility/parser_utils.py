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
import math
from collections import Counter
from functools import reduce
import operator
import multiprocessing as mp
import time
from transformers import AutoTokenizer

from utility.tokenizer import Tokenizer
from utility.bert_tokenizer import bert_tokenizer
from utility.greedy_hitman import greedy_hitman

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"


def load_dataset(path, framework, language=None):
    def condition(s, f, l):
        return ("framework" not in s or f == s["framework"]) and ("framework" in s or f in s["targets"]) and (l is None or s["language"] == l)

    data = {}
    with open(path, encoding="utf8") as f:
        for sentence in f.readlines():
            sentence = json.loads(sentence)
            if condition(sentence, framework, language):
                data[sentence["id"]] = sentence

                if framework == "amr":
                    sentence["input"] = sentence["input"].replace('  ', ' ')
                    sentence["input"] = bytes(sentence["input"], 'utf-8').decode('utf-8', 'ignore')

                if "nodes" not in sentence:
                    sentence["nodes"] = []

                for node in sentence["nodes"]:
                    if "properties" in node:
                        node["properties"] = {prop: node["values"][prop_i] for prop_i, prop in enumerate(node["properties"])}
                        del node["values"]
                    else:
                        node["properties"] = {}

                if "edges" not in sentence:
                    sentence["edges"] = []

    return data


def add_companion(data, path, language: str, tokenization_mode="aggresive"):
    if path is None:
        add_fake_companion(data, language, tokenization_mode=tokenization_mode)
        return

    companion = {}
    with open(path, encoding="utf8") as f:
        for line in f.readlines():
            example = json.loads(line)
            companion[example["id"]] = example

    for sentence in list(data.values()):
        if sentence["id"] not in companion:
            del data[sentence["id"]]
            print(f"WARNING: sentence {sentence['id']} not found in companion, it's omitted from the dataset")

    error_count = 0
    for l in companion.values():
        if l["id"] in data:
            if l["input"].replace(' ', '') != data[l["id"]]["input"].replace(' ', ''):
                print(f"WARNING: sentence {l['id']} not matching companion")
                print(f"original: {data[l['id']]['input']}")
                print(f"companion: {l['input']}")
                print(flush=True)
                del data[l["id"]]
                error_count += 1
                continue

            if language == "zho":
                offset = 0
                for n in l["nodes"]:
                    index = l["input"][offset:].find(n["label"])
                    start = offset + index
                    end = start + len(n["label"])

                    n["anchors"] = [{"from": start, "to": end}]
                    offset = end

            last_start, last_end = None, None
            for i, node in reversed(list(enumerate(l["nodes"]))):
                assert len(node["anchors"]) == 1
                start, end = node["anchors"][0]["from"], node["anchors"][0]["to"]

                if last_start is not None and end - 1 > last_start:
                    node["anchors"][0]["to"] = last_end
                    l["nodes"].pop(i + 1)

                last_start, last_end = start, end

            data[l["id"]]["sentence"] = data[l["id"]]["input"]

            tokens = []
            for n in l["nodes"]:
                assert len(n["anchors"]) == 1
                tokens.append(l["input"][n["anchors"][0]["from"] : n["anchors"][0]["to"]])

            if ''.join(tokens).replace(' ', '') != l["input"].replace(' ', '').replace(' ', '').replace(' ', ''):
                print(f"WARNING: sentence {l['id']} not matching companion after tokenization")
                print(f"companion input: {l['input']}")
                print(f"original: {data[l['id']]['input']}")
                print(f"tokens: {tokens}")
                print(flush=True)
                del data[l["id"]]
                error_count += 1
                continue

            data[l["id"]]["input"] = tokens

    for sentence in list(data.values()):
        try:
            create_token_anchors(sentence)
        except:
            print(f"WARNING: sentence {sentence['id']} not matching companion after anchor computation")
            print(f"tokens: {sentence['input']}")
            print(f"sentence: {sentence['sentence']}")
            print(flush=True)
            del data[sentence["id"]]
            error_count += 1

    print(f"{error_count} erroneously matched sentences with companion")


def add_fake_companion(data, language, tokenization_mode="aggresive"):
    tokenizer = Tokenizer(data.values(), mode=tokenization_mode)

    for sample in list(data.values()):
        sample["sentence"] = sample["input"]

        token_objects = tokenizer.create_tokens(sample)
        token_objects = [t for t in token_objects if t["token"] is not None]

        tokens = [t["token"]["word"] if isinstance(t["token"], dict) else t["token"] for t in token_objects]
        spans = [t["span"] for t in token_objects]

        sample["input"] = tokens
        sample["token anchors"] = spans


def create_token_anchors(sentence):
    offset = 0
    sentence["token anchors"] = []

    for w in sentence["input"]:
        spaces = 0
        index = sentence["sentence"][offset:].find(w)
        if index != 0 and (index < 0 or not sentence["sentence"][offset:offset + index].isspace()) and offset < len(sentence["sentence"]):
            while offset < len(sentence["sentence"]) and sentence["sentence"][offset] == ' ':
                offset += 1

            index = sentence["sentence"][offset:].replace(' ', '', 1).find(w)
            spaces = 1
            if index < 0:
                raise Exception(f"sentence {sentence['id']} not matching companion after anchor computation.")

        start = offset + index
        end = start + len(w) + spaces

        sentence["token anchors"].append({"from": start, "to": end})
        offset = end


def normalize_properties(data):
    for sentence in data.values():
        properties = []
        node_id = len(sentence["nodes"])
        for node in sentence["nodes"]:
            for relation, value in node["properties"].items():
                nodedized = {
                    "id": node_id,
                    "label": value,
                    "property": True,
                }
                if "anchors" in node:
                    nodedized["anchors"] = node["anchors"]
                properties.append(nodedized)
                sentence["edges"].append({"source": node["id"], "target": node_id, "label": relation, "property": True})

                node_id += 1

            del node["properties"]
        sentence["nodes"] += properties


def node_generator(data):
    for d in data.values():
        for n in d["nodes"]:
            yield n, d


def anchor_ids_from_intervals(data):
    for node, sentence in node_generator(data):
        if "anchors" not in node:
            node["anchors"] = []
        node["anchors"] = sorted(node["anchors"], key=lambda a: (a["from"], a["to"]))
        node["token references"] = set()

        for anchor in node["anchors"]:
            for i, token_anchor in enumerate(sentence["token anchors"]):
                if token_anchor["to"] <= anchor["from"]:
                    continue
                if token_anchor["from"] >= anchor["to"]:
                    break

                node["token references"].add(i)

        node["anchor intervals"] = node["anchors"]
        node["anchors"] = sorted(list(node["token references"]))
        del node["token references"]

    for sentence in data.values():
        sentence["token anchors"] = [[a["from"], a["to"]] for a in sentence["token anchors"]]


def tokenize(data, mode="aggressive"):
    tokenizer = Tokenizer(data.values(), mode=mode)
    for key in data.keys():
        data[key] = tokenizer(data[key])
        data[key] = tokenizer.clean(data[key])


def create_bert_tokens(data, encoder: str):
    tokenizer = AutoTokenizer.from_pretrained(encoder)

    for sentence in data.values():
        to_scatter, bert_input = bert_tokenizer(sentence, tokenizer, encoder)
        sentence["to scatter"] = to_scatter
        sentence["bert input"] = bert_input


def create_edges(sentence, label_f=None, normalize=False):
    N = len(sentence["nodes"])

    sentence["edge presence"] = [N, N, []]
    sentence["edge labels"] = [N, N, []]

    for e in sentence["edges"]:
        if normalize and "normal" in e:
            target, source = e["source"], e["target"]
            label = e["normal"].lower() if "normal" in e else "none"
        else:
            source, target = e["source"], e["target"]
            label = e["label"].lower() if "label" in e else "none"

        if label_f is not None:
            label = label_f(label)

        sentence["edge presence"][-1].append((source, target, 1))
        sentence["edge labels"][-1].append((source, target, label))

    edge_counter = len(sentence["edge presence"][-1])
    return edge_counter
