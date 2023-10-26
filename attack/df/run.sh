#!/bin/bash


# WF attack
WF=df
# data directory
d_dir=regulator/goodenough/standard
# classes
class=2
# classication results
res=regulator-ge-standard

echo "-------- [$WF] extracting features --------"
  #./extract.py --attack $WF --data_dir $d_dir
echo "-------- [$WF] classification --------"  
  ./classify.py --in $d_dir --out $res --class $class
   
#for i in {1..10}
#do
#  echo "-------- ${i}th classification --------"
#  ./classify.py --in $data_dir/feature.pkl --out $result
#done 

echo "----------  all done   ----------"
