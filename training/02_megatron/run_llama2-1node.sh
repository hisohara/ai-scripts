#!/bin/bash

export ROCM_PATH=/opt/rocm

# For TransformerEngine
# Not sure whether it is still needed for the execution
export NVTE_FRAMEWORK=pytorch
export PYTORCH_ROCM_ARCH=gfx942
export NVTE_USE_HIPBLASLT=1

export DATA_PATH="data/data_text_document"
export "TOKENIZER_MODEL=NousResearch/Llama-2-7b-chat-hf"

TEE_OUTPUT=1 \
MBS=4 \
BS=256 \
TP=1 \
TE_FP8=1 \
SEQ_LENGTH=4096 \
MODEL_SIZE=7 \
TOTAL_ITERS=50 \
USE_FLASH_ATTN=1 \
GEMM_TUNING=1 \
bash examples/llama/train_llama2.sh
