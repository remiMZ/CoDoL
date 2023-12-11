#!/bin/bash

# custom config
DATA="/path/to/dataset/folder"
TRAINER=ZeroshotCLIP

DATASET=$1
CFG=$2  
DATASET_OOD=$3

DIR=outputs/${DATASET}/${CFG}/${TRAINER}/${DATASET_OOD}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Resuming"

    python train.py \
    --root ${DATA} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/ood/${DATASET_OOD}.yaml \
    --config-file configs/trainers/zsclip/${CFG}.yaml \
    --output-dir ${DIR}\
    --eval-only
else
    echo "Run this job and save the output to ${DIR}"
    
    python train.py \
    --root ${DATA} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/ood/${DATASET_OOD}.yaml \
    --config-file configs/trainers/zsclip/${CFG}.yaml \
    --output-dir ${DIR}\
    --eval-only 
fi