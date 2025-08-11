#!/bin/bash
set -euo pipefail

# dataset
DATASET="wikitext-2-raw-v1"

# model list
MODELS=(
  "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)

# baseline evaluation: fp16, fp32
#BASELINE_DT_LIST=(fp16)

# quantization configuration
WQ_BIT_LIST=(3 5 6 7)
WQ_GROUPSIZE_LIST=(128 -1)

for MODEL in "${MODELS[@]}"; do
  echo
  echo "======================================================================"
  echo " Model: $MODEL   Dataset: $DATASET"
  echo "======================================================================"

  # ─── Baseline evaluation ───────────────────────────────────────────────
  #for BASE_DT in "${BASELINE_DT_LIST[@]}"; do
  #  echo
  #  echo "  [Baseline] datatype=${BASE_DT}"
  #  python eval.py \
  #    --model_name "${MODEL}" \
  #    --dataset "${DATASET}" \
  #    --baseline_datatype "${BASE_DT}"
  #done

  # ─── Quantization evaluation ───────────────────────────────────────────
  for WQ_BIT in "${WQ_BIT_LIST[@]}"; do
    # skip if bits==16 or 32
    if [[ "$WQ_BIT" -eq 16 || "$WQ_BIT" -eq 32 ]]; then
      continue
    fi

    # datatype
    case $WQ_BIT in
      3)  DTYPE_LIST=(int3 int3_asym fp3)                   ;;
      4)  DTYPE_LIST=(int4 int4_asym fp4)                  ;;
      5)  DTYPE_LIST=(int5 int5_asym fp5_e3m1 fp5_e2m2)    ;; # (int5 int5_asym fp5_e3m1 fp5_e2m2) 
      6)  DTYPE_LIST=(int6 int6_asym fp6_e2m3 fp6_e3m2)    ;;
      7)  DTYPE_LIST=(int7 int7_asym)                      ;;
      8)  DTYPE_LIST=(fp8_e2m5 fp8_e3m4 fp8_e4m3 fp8_e5m2) ;; # (int8 int8_asym fp8_e2m5 fp8_e3m4 fp8_e4m3 fp8_e5m2)
      *)  echo "Unsupported bit width: ${WQ_BIT}" >&2; continue ;;
    esac

    for DTYPE in "${DTYPE_LIST[@]}"; do
      for GS in "${WQ_GROUPSIZE_LIST[@]}"; do
        echo
        echo "  [Quant] bits=${WQ_BIT}, datatype=${DTYPE}, groupsize=${GS}"
        python eval.py \
          --model_name "${MODEL}" \
          --dataset "${DATASET}" \
          --wq_bits "${WQ_BIT}" \
          --wq_datatype "${DTYPE}" \
          --wq_groupsize "${GS}"
      done
    done
  done
done
