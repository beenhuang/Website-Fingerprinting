#!/bin/bash

# trace directory
data=Wang-20000

feature=feature.pkl
result=wang


echo "--------  run Huang's CF --------"
  
  ./extract.py --in $data
  #./run_CF.py --in $data/$feature --out $result

echo "----------  all done   ----------"


