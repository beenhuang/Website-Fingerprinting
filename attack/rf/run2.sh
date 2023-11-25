#!/bin/bash


# data directory
d_dir=trafficsilver/bigenough/standard
# WF attack
WF=rf
# number of classification categories
class=2
# classication result
res=bestd-ts-rf-twoclass

for j in {0..4}
  do   
  for i in {1..10}
    do
      echo "-------- [$WF] ${i}th classification --------"
      ./extract/extract.py --in $d_dir-$i-$j --out $WF/feature/$d_dir-$i-$j 
      ./classify/open-world.py --in $d_dir-$i-$j --out $res --class $class
    done
  done
echo "---------- $i experiments completed! ----------"