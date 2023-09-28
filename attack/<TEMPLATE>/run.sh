#!/bin/bash

run=/home/shadow/paper3/attack/df/run.sh
# directory
dir=/home/shadow/paper3/attack/df
# WF attack
attack=df
# WF defense
defense=dfd
# data directory
data=Wang-20000
# feature pickle file
feature=feature.pkl

echo "-------- run $attack script --------"
  $dir/extract.py --in $defense/$data --attack $attack
  $dir/classify.py --in $defense/$data/$feature --out ${defense}-$data
   
#for i in {1..10}
#do
#  echo "-------- ${i}th classification --------"
#  ./classify.py --in $data/$feature --out $result
#done 

echo "----------  all done   ----------"

