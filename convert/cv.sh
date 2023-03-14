#!/bin/bash


# raw data directory
data=goodenough/standard

echo "-------- start to convert --------"

  ./classify.py --in $data

echo "----------  all done   ----------"