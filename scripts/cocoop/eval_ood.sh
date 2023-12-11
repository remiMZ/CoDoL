#!/bin/bash

# custom config
DATA="/path/to/dataset/folder"
TRAINER=CoCoOp

DATASET=$1
CFG=$2
DATASET_OOD=$3
NAME=$4
LOADEP=5


for SEED in 1 2 3
do
    COMMON_DIR=${DATASET}/${CFG}/${TRAINER}/NAME${NAME}/${DATASET_OOD}/seed${SEED}
    MODEL_DIR=outputs/${COMMON_DIR}
    DIR=outputs_test/${COMMON_DIR}

    if [ -d "$DIR" ]; then
        echo "Evaluating model"
        echo "Results are available in ${DIR}. Resuming..."

        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/ood/${DATASET_OOD}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --model-dir ${MODEL_DIR} \
        --load-epoch ${LOADEP} \
        --eval-only 
    else
        echo "Evaluating model"
        echo "Runing the first phase job and save the output to ${DIR}"

        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/ood/${DATASET_OOD}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --model-dir ${MODEL_DIR} \
        --load-epoch ${LOADEP} \
        --eval-only 
    fi
done