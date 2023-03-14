#!/bin/bash

# trace directory
data=Wang-20000
# feature pickle file
feature=feature.pkl
# classication results
result=original-wang-xg


echo "--------  run Huang's WF --------"
  
  #./extract.py --in $data
  ./classify.py --in $data/$feature --out $result

echo "----------  all done   ----------"

#for i in {1..10}
#do
#  echo "---------------     run for the $i time    ---------------"
#  ./classify.py --in $data/$feature --out $result-$i

#done

