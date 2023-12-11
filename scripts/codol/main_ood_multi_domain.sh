#!/bin/bash

# custom config
DATA="/path/to/dataset/folder"
TRAINER=CoDoL_multi_domain

DATASET=$1
CFG=$2  # config file
DATASET_OOD=$3
NAME=$4
N_CTX=$5
N_DMX=$6

for SEED in 1 2 3
do
    DIR=outputs/multi_domain/${DATASET}/${CFG}/${TRAINER}/NAME${NAME}_N_CTX${N_CTX}_N_DMX${N_DMX}/${DATASET_OOD}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Resuming"

        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/ood/${DATASET_OOD}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.DPTSPSINGLE.N_CTX ${N_CTX} \
        TRAINER.DPTSPSINGLE.N_DMX ${N_DMX}
    else
        echo "Run this job and save the output to ${DIR}"

        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/ood/${DATASET_OOD}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.DPTSPSINGLE.N_CTX ${N_CTX} \
        TRAINER.DPTSPSINGLE.N_DMX ${N_DMX}
    fi
done