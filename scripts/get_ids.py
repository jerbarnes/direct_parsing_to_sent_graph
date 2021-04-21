#!/usr/bin/env python
# coding: utf-8

import json


def get_ids(fn):
    ids = []
    with open(fn) as fh:
        for line in fh:
            jline = json.loads(line)
            ids.append(jline["id"])
    return ids


if __name__ == "__main__":
    datasets = ['ca', 'ds_services', 'ds_unis', 'en', 'es', 'eu', 'mpqa',
                'norec_fine']
    splits = ["train", "dev", "test"]
    path = lambda ds, sp, suffix: f"../data/{ds}/{sp}.{suffix}"
    for dataset in datasets:
        for split in splits:
            ids = get_ids(path(dataset, split, "mrp"))
            with open(path(dataset, split, "ids"), "w") as fh:
                for i in ids:
                    print(i, file=fh)
