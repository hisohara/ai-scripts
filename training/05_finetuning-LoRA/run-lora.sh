#!/bin/bash

set -euo pipefail

NUMS="500 5000 50000 500000 1000000"
EVALS="jamcqa jnli jcommonsenseqa jcommonsenseqa_mc jsquad aio xlsum_ja"
#NUMS="500 1000"
#EVALS="jnli jcommonsenseqa"

WORKDIR=$PWD
SUM_LOG=1.SUMMARY
RESULT_LOG=1.RESULT

for NUM in $NUMS; do
  echo "Start finetune with ${NUM} samples at  [$(date '+%Y-%m-%d %H:%M:%S %z')]" >> $SUM_LOG
  python finetune-lora.py $NUM
  echo "Finish finetune with ${NUM} samples at [$(date '+%Y-%m-%d %H:%M:%S %z')]" >> $SUM_LOG

  MODEL_PATH="${WORKDIR}/qwen2-7b-llmjp-lora-${NUM}-merged"

  for EVAL in $EVALS; do
    OUTPUT_DIR="results/${EVAL}-${NUM}-qwen2-7b"

    echo "  Flexeval start with ${EVAL} at  [$(date '+%Y-%m-%d %H:%M:%S %z')]" >> $SUM_LOG
    flexeval_lm \
      --language_model HuggingFaceLM \
      --language_model.model "Qwen/Qwen2-7B-Instruct" \
      --language_model.default_gen_kwargs "{ do_sample: false }" \
      --force true \
      --eval_setup $EVAL \
      --save_dir "${OUTPUT_DIR}-bare"

    flexeval_lm \
      --language_model HuggingFaceLM \
      --language_model.model ${MODEL_PATH} \
      --language_model.default_gen_kwargs "{ do_sample: false }" \
      --force true \
      --eval_setup $EVAL \
      --save_dir "${OUTPUT_DIR}-merged"
    echo "  Flexeval finish with ${EVAL} at [$(date '+%Y-%m-%d %H:%M:%S %z')]" >> $SUM_LOG

    {
       printf '%s\n' "${OUTPUT_DIR}-bare/metrics.json"
       cat "${OUTPUT_DIR}-bare/metrics.json"
       printf '\n%s\n' "${OUTPUT_DIR}-merged/metrics.json"
       cat "${OUTPUT_DIR}-merged/metrics.json"
       printf '\n\n'
    } | tee -a "$RESULT_LOG"
  done
done
