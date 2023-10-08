#!/bin/bash


data=goodenough/safest
wfd=tamaraw


echo "--------  run $wfd --------"
  
  ./main.py --in $data --wfd $wfd

echo "----------  all done   ----------"