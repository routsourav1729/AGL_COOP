DATA=/raid/biplab/datasets/sourav/data
TRAINER=AGCoCoOpTrainer
DATASET=$1
SEED=$2
GPU_ID=${3:-3}  # Default to GPU 3 if not specified # Default to GPU 0 if not specified

echo "Using GPU: ${GPU_ID}"
CFG=vit_b16_c16_ep10_batch1
SHOTS=16

DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
else
    # Use only one GPU to avoid DataParallel issues
    CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --gpu ${GPU_ID} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base
fi