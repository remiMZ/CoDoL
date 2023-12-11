#!/bin/bash

# custom config
DATA="/path/to/dataset/folder"
TRAINER=VPT

DATASET=$1
CFG=$2  # config file
DATASET_OOD=$3
NAME=$4

for SEED in 1 2 3
do
    DIR=outputs/${DATASET}/${CFG}/${TRAINER}/NAME${NAME}/${DATASET_OOD}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Resuming..."
        
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/ood/${DATASET_OOD}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} 
    else
        echo "Run this job and save the output to ${DIR}"

        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/ood/${DATASET_OOD}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} 
    fi
done