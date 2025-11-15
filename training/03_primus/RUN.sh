#!/bin/bash

export DOCKER_IMAGE=rocm/primus:v25.9_gfx942
export HF_TOKEN=<YOUR TOKEN>
export HSA_NO_SCRATCH_RECLAIM=1
export NVTE_CK_USES_BWD_V3=1
export NCCL_IB_HCA=bnxt_re0:1,bnxt_re1:1,bnxt_re2:1,bnxt_re3:1,bnxt_re4:1,bnxt_re5:1,bnxt_re7:1,bnxt_re8:1 # specify which RDMA interfaces to use for communication
export NCCL_SOCKET_IFNAME=enp49s0f1np1 # your Network Interface
export GLOO_SOCKET_IFNAME=enp49s0f1np1 # your Network Interface
export NCCL_IB_GID_INDEX=3 # Set InfiniBand GID index for NCCL communication. Default is 3 for ROCE
export CPUS_PER_TASK=256
export CLEAN_DOCKER_CONTAINER=1
export REBUILD_BNXT=1
export PATH_TO_BNXT_TAR_PACKAGE=/shared/amdgpu/home/hisaki_ohara_7kq/libbnxt_re-231.0.162.0.tar.gz

## Llama2 70B
#NNODES=2 \
#EXP=examples/megatron/configs/llama2_70B-pretrain.yaml \
#bash examples/run_slurm_pretrain.sh \
#    --micro_batch_size 10 \
#    --global_batch_size 640 \
#    --recompute_num_layers 80 \
#    --no_fp8_weight_transpose_cache true \
#    --fp8 hybrid

## Llama2 7B
#NNODES=8 \
#EXP=examples/megatron/configs/llama2_7B-pretrain.yaml \
#bash ./examples/run_slurm_pretrain.sh \
#    --global_batch_size 2048 \
#    --fp8 hybrid

# Mixtral 8x7B
NNODES=8 \
EXP=examples/megatron/configs/mixtral_8x7B_v0.1-pretrain.yaml \
bash examples/run_slurm_pretrain.sh \
    --micro_batch_size 2 \
    --global_batch_size 256
