#!/bin/bash

python3 ../mtool/main.py --node_centric --strings --ids --read mrp --write norec "$1" "$1_converted"
# python3 ../evaluation/evaluate_single_dataset.py $2 "$1_converted"
