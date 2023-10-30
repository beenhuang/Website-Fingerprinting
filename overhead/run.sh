#!/bin/bash


un_dir=goodenough/standard
def=regulator
out=regulator-ge-standard

echo "--------  run --------"

  ./main.py --undefended_dir $un_dir --WF_defense $def --out $out
  
echo "----------  all done   ----------"