#!/bin/bash

for dataset in ca ds_services ds_unis en es eu mpqa norec_fine;
do
    for split in train dev test;
    do
        mkdir ../data/$dataset/dot;
        mkdir ../data/$dataset/dot/$split;
        mkdir ../data/$dataset/svg;
        mkdir ../data/$dataset/svg/$split;
        while IFS='' read -r LINE || [ -n "${LINE}" ];
        #for i in $(cat ../data/$dataset/$split.ids);
        do
            i=$LINE
            echo $i;
            k=$(echo $i | sed "s/\ /_/g")
            j=$(basename $k);
            # delete the --strings flag to show spans instead of tokens 
            python ../mtool/main.py --read mrp \
            --id "${i}" --ids --strings --write dot \
            ../data/$dataset/$split.mrp ../data/$dataset/dot/$split/$j.dot;
        #done
        done < ../data/$dataset/$split.ids
        for i in $(find ../data/$dataset/dot/$split -size 0);
        do
            /bin/rm -f $i;
        done;
        for i in ../data/$dataset/dot/$split/*.dot;
        do
          j=$(basename $i .dot);
          dot -Tsvg $i > ../data/$dataset/svg/$split/${j}.svg;
        done
    done
done
