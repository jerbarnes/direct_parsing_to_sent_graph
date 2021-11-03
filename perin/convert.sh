#!/bin/bash

python3 ../mtool/main.py --node_centric --strings --ids --read mrp --write norec "$1" "$1_converted"
