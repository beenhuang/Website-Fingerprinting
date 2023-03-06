#!/bin/bash

# trace directory
data=Wang-20000

feature=feature.pkl
result=wang

echo "-------- CUMUL evaluation --------"
  ./exfeature.py --in $data
  ./run_cumul.py --in $data/$feature --out $result

echo "----------  all done   ----------"
