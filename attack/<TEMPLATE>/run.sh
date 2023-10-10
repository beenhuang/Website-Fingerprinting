#!/bin/bash


# WF attack
WF=tiktok
# data directory
d_dir=regulator/goodenough/standard
# classes
class = 51
# classication results
res=regulator-ge-standard

echo "-------- run $WF --------"
  ./extract.py --attack $WF --data_dir $d_dir
  #./classify.py --in $data_dir/feature.pkl --out $res --class $class
   
#for i in {1..10}
#do
#  echo "-------- ${i}th classification --------"
#  ./classify.py --in $data_dir/feature.pkl --out $result
#done 

echo "----------  all done   ----------"

