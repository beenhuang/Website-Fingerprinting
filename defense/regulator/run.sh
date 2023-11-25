#!/bin/bash


d_dir=bigenough/safest
defense=regulator


for i in {1..10}
  do
    echo "-------- [$defense] ${i}th experiment --------"
    ./main.py --in $d_dir --out $defense/$d_dir-$i
  done

echo "---------- $i experiments completed! ----------"
