#!/bin/bash

# custom config
DATA="/path/to/dataset/folder"
TRAINER=CoOp

DATASET=$1
CFG=$2  # config file
DATASET_OOD=$3
NAME=$4

CTP=end  # class token position (end or middle)
NCTX=16  # number of context tokens
CSC=False  # class-specific context (False or True)

for SEED in 1 2 3
do
    DIR=outputs/${DATASET}/${CFG}/${TRAINER}/NAME${NAME}/${DATASET_OOD}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Resuming"
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/ood/${DATASET_OOD}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} 
    else
        echo "Run this job and save the output to ${DIR}"
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/ood/${DATASET_OOD}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} 
    fi
done