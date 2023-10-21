#!/bin/bash


# WF attack
WF=awf
# data directory
d_dir=goodenough/standard
# classes
class=2
# classication results
res=ge-standard

echo "-------- [$WF] extracting features --------"
  #./extract.py --attack $WF --data_dir $d_dir
echo "-------- [$WF] classification --------"  
  ./classify.py --in $d_dir --out $res --class $class --load_model
   
#for i in {1..10}
#do
#  echo "-------- ${i}th classification --------"
#  ./classify.py --in $data_dir/feature.pkl --out $result
#done 

echo "----------  all done   ----------"
