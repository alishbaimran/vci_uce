#!/bin/bash
#SBATCH --job-name=vci_train
#SBATCH --partition=vci_gpu_priority
#SBATCH --gpus=1
#SBATCH --output=logs/vci_train_%j.log
#SBATCH --time=5-00:00:00
#SBATCH --mem=100G

source ~/.bashrc
conda activate vci-f

MODEL_RUN_NAME="UCE_plain"
CHECKPOINT_DIR="/scratch/ctc/ML/uce/model_checkpoints/${MODEL_RUN_NAME}_${SLURM_JOB_ID}"

mkdir -p ${CHECKPOINT_DIR}
echo "Running with Checkpoint Directory: ${CHECKPOINT_DIR}"

export TRITON_CACHE_DIR="/home/alishbaimran/triton"
mkdir -p ${TRITON_CACHE_DIR}

python train_lit.py \
    --sample_size=1024 \
    --nlayers=4 \
    --pad_length=1425 \
    --compiled \
    --emsize=512 \
    --d_hid=2048 \
    --max_lr=0.0003 \
    --output_dim=512 \
    --n_epochs=8 \
    --batch_size=24 \
    --token_location=/scratch/ctc/ML/uce/all_species_pe_tokens.torch \
    --token_dim=5120 \
    --emb_model_name=ESM2_base \
    --dataset_path=/scratch/ctc/ML/uce/full_train_datasets.csv \
    --checkpoint_dir=${CHECKPOINT_DIR} \
    --run_name=${MODEL_RUN_NAME}_${SLURM_JOB_ID}
