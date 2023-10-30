#!/bin/bash


# WF attack
WF=tiktok
# data directory
d_dir=trafficsilver/goodenough/safest/0
# classes
class=2
# classication results
res=trafficsilver-ge-safest-0

echo "-------- [$WF] extracting features --------"
  ./extract.py --attack $WF --data_dir $d_dir
echo "-------- [$WF] classification --------"  
  ./classify.py --in $d_dir --out $res --class $class
   
#for i in {1..10}
#do
#  echo "-------- ${i}th classification --------"
#  ./classify.py --in $data_dir/feature.pkl --out $result
#done 

echo "----------  all done   ----------"
