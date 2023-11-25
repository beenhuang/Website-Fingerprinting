#!/bin/bash


# data directory
d_dir=hywf/bigenough/standard
# WF attack
WF=tiktok
# number of classification categories
class=2
# classication result
res=bigstd-hywf-tiktok-twoclass

for j in {0..1}
  do   
  for i in {1..10}
    do
      echo "-------- [$WF] ${i}th classification --------"
      ./extract/extract.py --in $d_dir-$i-$j --out $WF/feature/$d_dir-$i-$j 
      ./classify/open-world.py --in $d_dir-$i-$j --out $res --class $class
    done
  done
echo "---------- $i experiments completed! ----------"