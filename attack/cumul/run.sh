#!/bin/bash


# data directory
d_dir=front/bigenough/standard
# WF attack
WF=cumul
# number of classification categories
class=2
# classication result
res=bigstd-front-cumul-twoclass


for i in {1..1}
  do
    echo "-------- [$WF] ${i}th classification --------"
    ./extract/extract.py --in $d_dir-$i --out $WF/feature/$d_dir-$i 
    ./classify/open-world.py --in $d_dir-$i --out $res --class $class
  done

echo "---------- $i experiments completed! ----------"