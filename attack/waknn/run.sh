#!/bin/bash


# WF attack
WF=waknn
# data directory
d_dir=Wang-20000
# classes
class=101
# classication results
res=wang-20000

echo "-------- [$WF] extracting features --------"
  #./extract.py --attack $WF --data_dir $d_dir
echo "-------- [$WF] classification --------"  
  ./classify.py --in $d_dir --out $res --class $class
   
#for i in {1..10}
#do
#  echo "-------- [$WF] ${i}th classification --------"
#  ./extract.py --attack $WF --data_dir $d_dir
#  ./classify.py --in $d_dir --out $res --class $class
#done 

echo "----------  all done   ----------"

# compile attack
# g++ flearner.cpp -o flearner
