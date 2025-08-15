#!/bin/bash

export ROCM_PATH=/opt/rocm

# For TransformerEngine
# Not sure whether it is still needed for the execution
export NVTE_FRAMEWORK=pytorch
export PYTORCH_ROCM_ARCH=gfx942
export NVTE_USE_HIPBLASLT=1

export DATA_PATH="data/data_text_document"
export TOKENIZER_MODEL="NousResearch/Meta-Llama-3.1-70B"

CKPT_FORMAT=torch_dist \
TEE_OUTPUT=1 \
MBS=3 \
BS=24 \
TP=1 \
TE_FP8=0 \
FSDP=1 \
RECOMPUTE=1 \
SEQ_LENGTH=8192 \
MODEL_SIZE=70 \
TOTAL_ITERS=50 \
bash examples/llama/train_llama3.sh
