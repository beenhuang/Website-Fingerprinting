#!/bin/bash

run=/home/shadow/paper3/attack/df/run.sh
# directory
dir=/home/shadow/paper3/attack/df
# WF defense
defense=dfd
# feature pickle file
feature=feature.pkl

WF=df
dir=bigenough/standard
ul=100


echo "-------- run $attack script --------"
  ./extract.py --wf $WF --data_dir $dir --unmon_label $ul
  #$dir/classify.py --in $defense/$data/$feature --out ${defense}-$data
   
#for i in {1..10}
#do
#  echo "-------- ${i}th classification --------"
#  ./classify.py --in $data/$feature --out $result
#done 

echo "----------  all done   ----------"

