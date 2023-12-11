#!/bin/bash

# custom config
DATA="/path/to/dataset/folder"
TRAINER=CoOp

DATASET=$1
CFG=$2
DATASET_OOD=$3
NAME=$4

CTP=end  # class token position (end or middle)
NCTX=16  # number of context tokens
CSC=False  # class-specific context (False or True)
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
        --config-file /configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --model-dir ${MODEL_DIR} \
        --load-epoch ${LOADEP} \
        --eval-only \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} 
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
        --eval-only \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} 
    fi
done