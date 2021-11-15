DIR_A="predictions/comparable_dependency"
DIR_B="../sentiment_graphs/experiments"
DIR_P="predictions"
B="1e6"

LANGS="norec multibooked_eu multibooked_ca mpqa darmstadt_unis"
# LANGS="mpqa darmstadt_unis"

PERINS="node-centric labeled-edge opinion-tuple frozen_opinion-tuple"

# for lang in $LANGS:
# do
#     echo $lang
#     python evaluation/bootstrap.py \
#         --gold_file data/raw/$lang/test.json \
#         --pred_file_a $DIR_A/$lang/1/test.pred.json \
#                       $DIR_A/$lang/2/test.pred.json \
#                       $DIR_A/$lang/3/test.pred.json \
#                       $DIR_A/$lang/4/test.pred.json \
#                       $DIR_A/$lang/5/test.pred.json \
#         --pred_file_b $DIR_B/$lang/head_final/1/test.json.pred \
#                       $DIR_B/$lang/head_final/2/test.json.pred  \
#                       $DIR_B/$lang/head_final/3/test.json.pred \
#                       $DIR_B/$lang/head_final/4/test.json.pred \
#                       $DIR_B/$lang/head_final/5/test.json.pred \
#         --b $B \
#         --debug;
# 
# done


a="comparable_dependency"
b="frozen_opinion-tuple"
for lang in $LANGS;
do
    echo $lang
    python evaluation/bootstrap.py \
        --gold_file data/raw/$lang/test.json \
        --pred_file_a $DIR_P/$a/$lang/1/test.pred.json \
                      $DIR_P/$a/$lang/2/test.pred.json \
                      $DIR_P/$a/$lang/3/test.pred.json \
                      $DIR_P/$a/$lang/4/test.pred.json \
                      $DIR_P/$a/$lang/5/test.pred.json \
        --pred_file_b $DIR_P/$b/$lang/1/test.pred.json \
                      $DIR_P/$b/$lang/2/test.pred.json  \
                      $DIR_P/$b/$lang/3/test.pred.json \
                      $DIR_P/$b/$lang/4/test.pred.json \
                      $DIR_P/$b/$lang/5/test.pred.json \
        --b $B \
        --debug \
        --together;
done


for lang in $LANGS;
do
    echo $lang
    for a in $PERINS;
    do
        for b in $PERINS;
        do
            if [ $a = $b ];
            then
                #echo "$a = $b"
                break
            else
                echo $a $b
               python evaluation/bootstrap.py \
                   --gold_file data/raw/$lang/test.json \
                   --pred_file_a $DIR_P/$a/$lang/1/test.pred.json \
                                 $DIR_P/$a/$lang/2/test.pred.json \
                                 $DIR_P/$a/$lang/3/test.pred.json \
                                 $DIR_P/$a/$lang/4/test.pred.json \
                                 $DIR_P/$a/$lang/5/test.pred.json \
                   --pred_file_b $DIR_P/$b/$lang/1/test.pred.json \
                                 $DIR_P/$b/$lang/2/test.pred.json  \
                                 $DIR_P/$b/$lang/3/test.pred.json \
                                 $DIR_P/$b/$lang/4/test.pred.json \
                                 $DIR_P/$b/$lang/5/test.pred.json \
                   --b $B \
                   --debug \
                   --together;
            fi
        done
    done
done


# test all combinations of runs and count between those
B="1e5"

a="comparable_dependency"
b="frozen_opinion-tuple"
for lang in $LANGS;
do
    echo $lang
    python evaluation/bootstrap.py \
        --gold_file data/raw/$lang/test.json \
        --pred_file_a $DIR_P/$a/$lang/1/test.pred.json \
                      $DIR_P/$a/$lang/2/test.pred.json \
                      $DIR_P/$a/$lang/3/test.pred.json \
                      $DIR_P/$a/$lang/4/test.pred.json \
                      $DIR_P/$a/$lang/5/test.pred.json \
        --pred_file_b $DIR_P/$b/$lang/1/test.pred.json \
                      $DIR_P/$b/$lang/2/test.pred.json  \
                      $DIR_P/$b/$lang/3/test.pred.json \
                      $DIR_P/$b/$lang/4/test.pred.json \
                      $DIR_P/$b/$lang/5/test.pred.json \
        --b $B \
        --debug;
done


for lang in $LANGS;
do
    echo $lang
    for a in $PERINS;
    do
        for b in $PERINS;
        do
            if [ $a = $b ];
            then
                #echo "$a = $b"
                break
            else
                echo $a $b
               python evaluation/bootstrap.py \
                   --gold_file data/raw/$lang/test.json \
                   --pred_file_a $DIR_P/$a/$lang/1/test.pred.json \
                                 $DIR_P/$a/$lang/2/test.pred.json \
                                 $DIR_P/$a/$lang/3/test.pred.json \
                                 $DIR_P/$a/$lang/4/test.pred.json \
                                 $DIR_P/$a/$lang/5/test.pred.json \
                   --pred_file_b $DIR_P/$b/$lang/1/test.pred.json \
                                 $DIR_P/$b/$lang/2/test.pred.json  \
                                 $DIR_P/$b/$lang/3/test.pred.json \
                                 $DIR_P/$b/$lang/4/test.pred.json \
                                 $DIR_P/$b/$lang/5/test.pred.json \
                   --b $B \
                   --debug;
            fi
        done
    done
done