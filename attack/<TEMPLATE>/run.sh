#!/bin/bash

# WF attack
attack=Var-CNN
# trace directory
data=Wang-20000
# feature pickle file
feature=feature.pkl
# classication results
result=wang


echo "-------- run $attack --------"
  ./extract.py --in $data --attack $attack
  ./classify.py --in $data/$feature --out $result

echo "----------  all done   ----------"
