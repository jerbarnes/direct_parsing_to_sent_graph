#!/usr/bin/env python3
# conding=utf-8
#
# Copyright 2020 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from data.parser.from_mrp.abstract_parser import AbstractParser
import utility.parser_utils as utils


class EvaluationParser(AbstractParser):
    def __init__(self, args, fields):
        path = args.test_data
        self.data = utils.load_dataset(path)
        utils.tokenize(self.data, mode="space")

        for sentence in self.data.values():
            sentence["token anchors"] = [[a["from"], a["to"]] for a in sentence["token anchors"]]

        utils.create_bert_tokens(self.data, args.encoder)

        super(EvaluationParser, self).__init__(fields, self.data)
