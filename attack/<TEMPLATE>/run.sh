#!/bin/bash

# WF attack
attack=tiktok
# trace directory
data=Wang-20000
# feature pickle file
feature=feature.pkl
# classication results
result=Wang-20000

echo "-------- run $attack --------"
  #echo "-------- extract features for $attack --------"
  #./extract.py --in $data --attack $attack
  #./classify.py --in $data/$feature --out $result
  
#for i in {1..10}
#do
  #echo "-------- classify $i times for $attack --------"
  #./classify.py --in $data/$feature --out $result
#done 

echo "----------  all done   ----------"

