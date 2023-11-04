#!/bin/bash


# WF attack
WF=cumul
# data directory
d_dir=bigenough/standard
# classes
class=96
# classication results
res=be-standard

echo "-------- [$WF] extracting features --------"
  ./extract.py --attack $WF --data_dir $d_dir
echo "-------- [$WF] classification --------"  
  ./classify.py --in $d_dir --out $res --class $class --find_params
   
#for i in {1..10}
#do
#  echo "-------- [$WF] ${i}th classification --------"
#  ./extract.py --attack $WF --data_dir $d_dir
#  ./classify.py --in $d_dir --out $res --class $class
#done 

echo "----------  all done   ----------"
