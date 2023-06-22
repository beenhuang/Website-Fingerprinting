#!/bin/bash

# trace directory
data=Wang-20000
defense=FRONT


echo "--------  run $defense --------"
  
  ./main.py --in $data --defense $defense

echo "----------  all done   ----------"


