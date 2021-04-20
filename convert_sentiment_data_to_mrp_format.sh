#!/bin/bash

for dataset in ca ds_services ds_unis en es eu mpqa norec_fine; do
    for split in train dev test; do
        indata=data/"$dataset"/"$split".json;
        outdata=data/"$dataset"/"$split".mrp;
        #python3 mtool/main.py --reify --strings --ids --read norec --write mrp "$indata" "$outdata"
        python3 mtool/main.py --reify --strings --ids --read norec --write mrp "$indata" "$outdata"
    done;
done;
