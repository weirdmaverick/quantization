#!/bin/bash
set -euo pipefail

if [ $# -lt 4 ]; then
  echo "Usage: $0 BITWIDTH TENSOR_NAME THRESHOLD FORMAT1 [FORMAT2 ...]"
  echo "Example: $0 4 model.decoder.layers.0.self_attn.k_proj.weight 0.01 int4_asym fp4"
  exit 1
fi

BIT=$1; TENSOR=$2; TH=$3; shift 3
FORMATS=("$@")

BASEDIR="../onnx_graph/facebook__opt-1.3b"
BASE="$BASEDIR/fp16_baseline/model.onnx"

# build -q args
QARGS=()
for fmt in "${FORMATS[@]}"; do
  PATH="$BASEDIR/per-channel/${fmt}/w_${BIT}_gs_none/model.onnx"
  QARGS+=( -q "$fmt" "$PATH" )
done

python compare.py \
  -b "$BASE" \
  "${QARGS[@]}" \
  -t "$TENSOR" \
  -th "$TH"


