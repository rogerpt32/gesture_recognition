#!/bin/bash

DATA_DIR="../data/videos/"
TRAIN_LIST="../data/trainlist.txt"
GMM_OUT="../data/fishers/gmm_list"

python gmm.py 120 $DATA_DIR $TRAIN_LIST $GMM_OUT --pca

TEST_LIST="../data/testlist.txt"

training_output="../data/fishers/train/"
testing_output="../data/fishers/test/"

python computeFVs.py $DATA_DIR $TRAIN_LIST $training_output $GMM_OUT
python computeFVs.py $DATA_DIR $TEST_LIST $testing_output $GMM_OUT

CLASS_INDEX="../data/class_index.txt"
CLASS_INDEX_OUT="../data/class_index"
python compute_class_index.py $CLASS_INDEX $CLASS_INDEX_OUT

python classify_experiment.py
