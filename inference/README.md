# Inference
## vLLM on AAC14 (MI355X)
```bash
$ podman pull docker.io/vllm/vllm-openai-rocm:v0.19.1
$ podman run -it --device=/dev/dri --device=/dev/kfd --device=/dev/infiniband --device=/dev/infiniband/rdma_cm \
--network=host --ipc=host --group-add keep-groups -e HF_TOKEN=$HF_TOKEN \
--entrypoint /bin/bash docker.io/vllm/vllm-openai-rocm:v0.19.1
```

### Server
```bash
$ more RUN.sh
#!/bin/bash

export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export VLLM_ROCM_USE_AITER_RMSNORM=0

vllm serve moonshotai/Kimi-K2.5 \
    --tensor-parallel-size 4 \
    --mm-encoder-tp-mode data \
    --block-size=1 \
    --tool-call-parser kimi_k2 \
    --reasoning-parser kimi_k2 \
    --enable-auto-tool-choice \
    --enable-prefix-caching \
    --trust-remote-code
```

### Client for benchmark
```bash
$ more bench-concurrency.sh
#!/usr/bin/env bash
set -euo pipefail

BASE_URL="http://<Server IP address>:8000"
MODEL="moonshotai/Kimi-K2.5"
RESULT_DIR="/app/results/kimi_tp4_random_concurrency"

mkdir -p "${RESULT_DIR}"

for C in 1 2 4 8 16 32 64; do
  OUT="${RESULT_DIR}/concurrency_${C}.txt"

  echo "Running concurrency=${C}"

  vllm bench serve \
    --backend openai-chat \
    --base-url "${BASE_URL}" \
    --endpoint /v1/chat/completions \
    --model "${MODEL}" \
    --dataset-name random \
    --input-len 8192 \
    --output-len 1024 \
    --num-prompts $((C * 8)) \
    --request-rate inf \
    --max-concurrency "${C}" \
    --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 95,99 \
    --trust-remote-code \
    2>&1 | tee "${OUT}"
done
```
