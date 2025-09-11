#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="results_quant"

# model list
declare -a model_list=(
  "facebook/opt-1.3b" "microsoft/phi-2" "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)
# ("facebook/opt-1.3b" "microsoft/phi-2" "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# ───────────────── Baseline (fp16) ─────────────────
declare -a baseline_list=("fp16")    # "fp16" "bf16" "fp32"

for model in "${model_list[@]}"; do
  for baseline in "${baseline_list[@]}"; do
    echo "===================="
    echo "[Baseline] ${baseline} → ${model}"
    python export_and_quant.py \
      --model_name "${model}" \
      --baseline_datatype "${baseline}"
  done
done

# ───────────────── Quantization ─────────────────
# bit-width
declare -a wq_bit_list=(4 8)

# Granularity: -1=per-tensor, 0=per-channel, 128=per-group, group size(128)
declare -a wq_groupsize_list=(-1)

for model in "${model_list[@]}"; do
  for wq_bit in "${wq_bit_list[@]}"; do
    # datatype 
    case "${wq_bit}" in
      3)  datatype_list=(int3 int3_asym fp3) ;;
      4)  datatype_list=(int4 int4_asym fp4) ;;  # (int4 int4_asym fp4)
      5)  datatype_list=(int5 int5_asym fp5_e2m2 fp5_e3m1) ;;
      6)  datatype_list=(int6 int6_asym fp6_e2m3 fp6_e3m2) ;;
      8)  datatype_list=(int8 int8_asym fp8_e2m5 fp8_e3m4 fp8_e4m3 fp8_e5m2) ;; #(int8 int8_asym fp8_e2m5 fp8_e3m4 fp8_e4m3 fp8_e5m2)
      *)  echo "Unsupported bit width: ${wq_bit}" >&2; continue ;;
    esac

    for dtype in "${datatype_list[@]}"; do
      for gs in "${wq_groupsize_list[@]}"; do
        echo "--------------------"
        echo "[Quant] Model=${model}"
        echo "       Bits=${wq_bit}, Datatype=${dtype}, GroupSize=${gs}"
        python export_and_quant.py \
          --model_name "${model}" \
          --wq_bits ${wq_bit} \
          --wq_datatype "${dtype}" \
          --wq_groupsize ${gs}
      done
    done
  done
done
