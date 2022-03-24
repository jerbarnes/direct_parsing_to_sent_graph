#!/bin/bash

for dataset in darmstadt_unis mpqa multibooked_ca multibooked_eu norec opener_en opener_es; do
    mkdir data/labeled_edge_mrp/"$dataset"
    mkdir data/node_centric_mrp/"$dataset"
    for split in train dev test; do
        indata=data/raw/"$dataset"/"$split".json;

        outdata_edge=data/labeled_edge_mrp/"$dataset"/"$split".mrp;
        outdata_node=data/node_centric_mrp/"$dataset"/"$split".mrp;

        python3 mtool/main.py --node_centric --strings --ids --read norec --write mrp "$indata" "$outdata_node"
        python3 mtool/main.py --strings --ids --read norec --write mrp "$indata" "$outdata_edge"
    done;
done;
