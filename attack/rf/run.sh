#!/bin/bash

# WF attack
attack=rf
# WF defense
defense=dfd
# data directory
data=Wang-20000
# feature pickle file
feature=feature.pkl

echo "-------- run $attack script --------"
  ./extract.py --in $data --attack $attack
  #$dir/classify.py --in $defense/$data/$feature --out ${defense}-$data
   
#for i in {1..10}
#do
#  echo "-------- ${i}th classification --------"
#  ./classify.py --in $data/$feature --out $result
#done 

echo "----------  all done   ----------"

