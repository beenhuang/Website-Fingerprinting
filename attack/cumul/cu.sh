#!/bin/bash

# trace directory
data=tamaraw/Wang-20000
# feature pickle file
feature=feature.pkl
# classication results
result=tamaraw-wang

echo "-------- CUMUL evaluation --------"
  #./extract.py --in $data
  ./classify.py --in $data/$feature --out $result

echo "----------  all done   ----------"
