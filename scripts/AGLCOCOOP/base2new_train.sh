
DATA=/home/sourav/ALL_FILES/Thesis/CasPL/data
# TRAINER=CoCoOp
# TRAINER=CoOp
TRAINER=AGCoCoOpTrainer
DATASET=$1
SEED=$2

CFG=vit_b16_c16_ep10_batch1
# CFG=vit_b16_c4_ep10_batch1_ctxv1
# CFG=vit_b16_ctxv1  # uncomment this when TRAINER=CoOp
# CFG=vit_b16_ep50_ctxv1  # uncomment this when TRAINER=CoOp and DATASET=imagenet
SHOTS=16


DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
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
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base
fi