#!/bin/bash
set -euo pipefail

# ───────────────────────────────────────────────────────────────
# usage example:
# ./single_weight_quant_error.sh \
#   ../onnx_graph/facebook__opt-1.3b/fp16_baseline/model.onnx \
#   ../onnx_graph/facebook__opt-1.3b/per-group/fp4/w_4_gs_128/model.onnx \
#   model.decoder.layers.0.self_attn.k_proj.weight \
#   0.001 \
#   0.015 \
#   "Quantized Error FP4"
# ───────────────────────────────────────────────────────────────

python single_weight_quant_error.py \
  --baseline-path "${1:-../onnx_graph/facebook__opt-1.3b/fp16_baseline/model.onnx}" \
  --quant-path    "${2:-../onnx_graph/facebook__opt-1.3b/per-channel/int4/w_4_gs_none/model.onnx}" \
  --tensor-name   "${3:-model.decoder.layers.0.self_attn.k_proj.weight}" \
  --step-size     "${4:-0.002}" \
  --max-value     "${5:-0.03}" \
  --title         "${6:-Quantized Error INT4-sym (per-channel)}"
# if bit-width == 4
# --step-size = 0.002
# --max-value = 0.03
# else 
# --step-size = 0.0001
# --max-value = 0.001
