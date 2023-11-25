#!/bin/bash


# data directory
d_dir=dfd/bigenough/standard
# WF attack
WF=rf
# number of classification categories
class=2
# classication result
res=bigstd-dfd-rf-twoclass


for i in {1..10}
  do
    echo "-------- [$WF] ${i}th classification --------"
    ./extract/extract.py --in $d_dir-$i --out $WF/feature/$d_dir-$i 
    ./classify/open-world.py --in $d_dir-$i --out $res --class $class
  done

echo "---------- $i experiments completed! ----------"