#!/usr/bin/env bash
set -euo pipefail

# save directory
RESULT_DIR = "../multi_weight_result"
mkdir -p "${RESULT_DIR}"

# baseline 
BASELINE_DIR="../onnx_graph/facebook__opt-1.3b/fp16_baseline"
MODEL_ROOT="../onnx_graph/facebook__opt-1.3b"

# granularity, datatype list
GRANS=("per-channel" "per-group")
DTYPES=("int8" "int8_asym" "int4" "int4_asym" "fp4" "fp8_e3m4" "fp8_e4m3")

# per-group default group size
DEFAULT_GS=128

for GRAN in "${GRANS[@]}"; do
  for DTYPE in "${DTYPES[@]}"; do
    # bits: datatype 문자열에서 첫 숫자 추출
    BITS=$(echo "$DTYPE" | grep -o '[0-9]\+' | head -n1)
    if [[ "$GRAN" == "per-group" ]]; then
      GS=$DEFAULT_GS
    else
      GS=none
    fi

    QUANT_DIR="${MODEL_ROOT}/${GRAN}/${DTYPE}/w_${BITS}_gs_${GS}"
    OUT="${RESULT_DIR}/${DTYPE}_${GRAN}_errors.csv"

    if [[ ! -d "$QUANT_DIR" ]]; then
      echo "[Warning] skip: no directory -> $QUANT_DIR"
      continue
    fi

    echo "=== Generating: datatype=$DTYPE, granularity=$GRAN, gs=$GS ==="
    python quant_error_all_weight.py \
      --baseline_dir "$BASELINE_DIR" \
      --quant_dir    "$QUANT_DIR" \
      --output_csv   "$OUT"

    echo "  → Saved: $OUT"
    echo
  done
done

