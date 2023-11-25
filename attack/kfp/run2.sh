#!/bin/bash


# data directory
d_dir=hywf/bigenough/standard
# WF attack
WF=df
# number of classification categories
class=2
# classication result
res=bestd-hywf-df-twoclass

for j in {2..4}
do   
for i in {1..10}
  do
    echo "-------- [$WF] ${i}th classification --------"
    ./extract.py --in $d_dir-$i-$j --out $WF/feature/$d_dir-$i-$j 
    ./open-world.py --in $d_dir-$i-$j --out $res --class $class
  done
done
echo "---------- $i experiments completed! ----------"