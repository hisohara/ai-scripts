#!/bin/bash
#SBATCH --job-name=megatron-pyxis
#SBATCH --partition=256C8G1H_MI325X_Ubuntu22
#SBATCH --reservation=gpu-14_gpu-4_gpu-11_gpu-6_gpu-12_gpu-28_gpu-29_gpu-30_reservation
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -o Log/%x-%j.out

set -euo pipefail

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

CONTAINER_IMAGE=/shared/amdgpu/home/hisaki_ohara_7kq/Projects/Megatron/rocm+megatron-lm+v25.9_gfx942-bnxt.sqsh

export NCCL_IB_HCA=bnxt_re0:1,bnxt_re1:1,bnxt_re2:1,bnxt_re3:1,bnxt_re4:1,bnxt_re5:1,bnxt_re7:1,bnxt_re8:1
export NCCL_SOCKET_IFNAME=enp49s0f1np1
export GLOO_SOCKET_IFNAME=enp49s0f1np1
export NCCL_IB_GID_INDEX=3

export HF_TOKEN=<YOUR TOKEN>

srun \
  --container-image=$CONTAINER_IMAGE \
  --container-writable \
  --container-workdir="/workspace/Megatron-LM" \
  --container-mounts=/shared/amdgpu/home/hisaki_ohara_7kq/Projects/Megatron/podman_shared:/podman_shared \
  --container-env="NCCL_IB_HCA,NCCL_SOCKET_IFNAME,GLOO_SOCKET_IFNAME,NCCL_IB_GID_INDEX,HF_TOKEN,MASTER_ADDR" \
  bash -c '
    TOKENIZER_MODEL=meta-llama/Llama-3.1-70B \
    DATA_CACHE_PATH=/podman_shared/cache \
    FP8_WEIGHT_TRANSPOSE_CACHE=0 \
    CKPT_FORMAT=torch_dist \
    RECOMPUTE=1 \
    TEE_OUTPUT=1 \
    MBS=4 \
    BS=256 \
    FSDP=1 \
    TP=1 \
    TE_FP8=1 \
    SEQ_LENGTH=8192 \
    MODEL_SIZE=70 \
    MOCK_DATA=1 \
    MASTER_ADDR=$MASTER_ADDR \
    NNODES=$SLURM_NNODES \
    NODE_RANK=$SLURM_NODEID \
    bash examples/llama/train_llama3.sh
  '

## Llama 3.1 8B, FP8, MBS2, BS256
#  bash -c '
#    TOKENIZER_MODEL=meta-llama/Llama-3.1-8B \
#    DATA_CACHE_PATH=/podman_shared/cache \
#    TEE_OUTPUT=1 \
#    MBS=2 \
#    BS=256 \
#    TP=1 \
#    TE_FP8=1 \
#    SEQ_LENGTH=8192 \
#    MODEL_SIZE=8 \
#    MOCK_DATA=1 \
#    MASTER_ADDR=$MASTER_ADDR \
#    NNODES=$SLURM_NNODES \
#    NODE_RANK=$SLURM_NODEID \
#    bash examples/llama/train_llama3.sh
#  '
