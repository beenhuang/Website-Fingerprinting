#!/bin/bash

run=/home/shadow/paper3/attack/df/run.sh
# directory
dir=/home/shadow/paper3/attack/df
# WF defense
defense=dfd
# feature pickle file
feature=feature.pkl

WF=varcnn
dir=Wang-20000
ul=100
result=wang20000
feature=feature.pkl

echo "-------- run $attack script --------"
  #./extract.py --wf $WF --data_dir $dir --unmon_label $ul
  ./classify.py --in $dir/$feature --out $result
   
#for i in {1..10}
#do
#  echo "-------- ${i}th classification --------"
#  ./classify.py --in $data/$feature --out $result
#done 

echo "----------  all done   ----------"

