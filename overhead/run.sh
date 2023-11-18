#!/bin/bash


d_dir=bigenough/standard
defense=decoy
res=decoy

for i in {1..10}
  do
    echo "-------- [$defense] ${i}th experiment --------"
    ./main.py --undef_dir $d_dir --def_dir $defense/$d_dir-$i --out $res
  done

echo "----------  all done   ----------"