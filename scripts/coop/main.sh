#!/bin/bash

# custom config
DATA=/home/sourav/ALL_FILES/Thesis/CasPL/data
TRAINER=AGL

DATASET=$1
CFG=$2
CTP=$3  # class token position
NCTX=$4  # number of context tokens
SHOTS=$5  # number of shots
CSC=$6  # class-specific context

for SEED in 1 2 3
do
    DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.AGL.N_CTX ${NCTX} \
        TRAINER.AGL.CSC ${CSC} \
        TRAINER.AGL.PROMPT_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS}
    fi
done