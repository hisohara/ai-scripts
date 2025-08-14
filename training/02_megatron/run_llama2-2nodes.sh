#!/bin/bash

export ROCM_PATH=/opt/rocm

# For TransformerEngine
# Not sure whether it is still needed for the execution
export NVTE_FRAMEWORK=pytorch
export PYTORCH_ROCM_ARCH=gfx942
export NVTE_USE_HIPBLASLT=1

export DATA_PATH="data/data_text_document"
export "TOKENIZER_MODEL=NousResearch/Llama-2-7b-chat-hf"

export MASTER_ADDR=<NODE0 IP>
export NNODES=2
export NODE_RANK=0
# DATA_CACHE_PATH should be the shared mount for multiple nodes.
export DATA_CACHE_PATH=/podman_shared/cache
# Please check `rdma link`
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_9
export NCCL_SOCKET_IFNAME=bond0

TEE_OUTPUT=1 \
MBS=4 \
BS=512 \
TP=1 \
TE_FP8=1 \
SEQ_LENGTH=4096 \
MODEL_SIZE=7 \
TOTAL_ITERS=50 \
USE_FLASH_ATTN=1 \
GEMM_TUNING=1 \
bash examples/llama/train_llama2.sh
