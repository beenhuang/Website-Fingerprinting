#!/bin/bash

# trace directory
data=Wang-20000
# feature pickle file
feature=feature.pkl
# classication results
result=wang-20000

echo "-------- CUMUL evaluation --------"
  ./extract.py --in $data
  ./classify.py --in $data/$feature --out $result

echo "----------  all done   ----------"
