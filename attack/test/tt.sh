#!/bin/bash

# trace directory
data=Wang-20000
# feature pickle file
feature=feature.pkl
# classication results
result=wang20000-knn


echo "--------  run test --------"
  
  #./extract.py --in $data
  ./classify.py --in $data/$feature --out $result

echo "----------  all done   ----------"


