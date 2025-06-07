DATA=/raid/biplab/datasets/sourav/data
TRAINER=AGCoCoOpTrainer
DATASET=$1
SEED=$2
GPU_ID=${3:-3}  # Default to GPU 3 if not specified

echo "Using GPU: ${GPU_ID}"
CFG=vit_b16_c16_ep10_batch1
SHOTS=16
LOADEP=10
SUB=new

COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
MODEL_DIR=output/base2new/train_base/${COMMON_DIR}
DIR=output/base2new/test_${SUB}/${COMMON_DIR}
if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
else
    # Use specified GPU via environment variable
    CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/CoCoOp/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR} \
    --load-epoch ${LOADEP} \
    --eval-only \
    --gpu ${GPU_ID} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB}
fi





