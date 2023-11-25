#!/bin/bash


# data directory
d_dir=bigenough/standard
# WF attack
WF=awf
# number of classification categories
class=2
# classication result
res=bigstd-awf-twoclass

   
for i in {1..1}
  do
    echo "-------- [$WF] ${i}th classification --------"
    ./extract/extract.py --in $d_dir --out $WF/feature/$d_dir-$i 
    ./classify/open-world.py --in $d_dir-$i --out $res --class $class
  done

echo "---------- $i experiments completed! ----------"
