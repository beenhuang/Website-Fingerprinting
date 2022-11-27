#!/bin/bash


datadir=server

outfile=orignal


echo "---------------     run Kwon's CF     ---------------"
./run_CF.py --in $datadir --out $outfile

echo "----------  all done   ----------"


