#!/bin/bash


# WF attack
WF=tiktok
# data directory
d_dir=regulator/bigenough/standard
# classes
class=2
# classication results
res=bigstd-regulator-tiktok-twoclass
   
for i in {1..10}
  do
    echo "-------- [$WF] ${i}th classification --------"
    ./extract/extract.py --in $d_dir-$i --out $WF/feature/$d_dir-$i
    ./classify/open-world.py --in $d_dir-$i --out $res --class $class
    #./one-page.py --in $d_dir-$i-2 --out $res
    #./closed-world.py --in $d_dir-$i-2 --out $res --class $class
  done

echo "----------  all done   ----------"
