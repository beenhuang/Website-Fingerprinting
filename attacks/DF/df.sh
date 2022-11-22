### Run DF model
#
# 1. TRAINING CASE: train/save trained DF and save test results
# df.py --train -i <ds-*.pkl> -o <res-*.csv> -sm <df-*.pkl> \
#                   
# 2. TESTING CASE: load trained DF, test DF and save result
# df.py -i <ds-*.pkl> -lm <df-*.pkl> -o <res-*.csv> \ 
#           

# TRAINING:
./df-train.py --train --in ds-*.pkl --out res-*.csv --model df-*.pkl 

# TESTING:
#./df-train.py --in ds-*.pkl --out res-*.csv --model df-*.pkl 

