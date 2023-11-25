#!/bin/bash


d_dir=bigenough
defense=dfd

for m in standard safer safest
  do
  for i in {1..10}
    do
      echo "-------- [$defense] ${i}th experiment --------"
      ./main.py --in $d_dir/$m --out $defense/$d_dir/$m-${i}
    done
  done

echo "---------- ${i} experiments completed! ----------"
