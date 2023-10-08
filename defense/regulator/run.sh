#!/bin/bash


data=goodenough/safest
wfd=regulator


echo "--------  run $wfd --------"
  
  ./main.py --in $data --wfd $wfd

echo "----------  all done   ----------"