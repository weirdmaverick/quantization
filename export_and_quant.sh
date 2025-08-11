#!/bin/bash
set -euo pipefail

# data save directory 
export HF_HOME="results_quant"

# model
declare -a model_list=(
  "facebook/opt-1.3b"
  "microsoft/phi-2"
  "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)
  #"facebook/opt-1.3b"
  #"microsoft/phi-2"
  #"TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# bit width (baseline)
#declare -a baseline_list=(16)

# bit width (quant)
declare -a wq_bit_list=(5)

# group size: per-group(128), per-channel(-1)
declare -a wq_groupsize_list=(128 -1)

for model in "${model_list[@]}"; do

  # ─── Baseline ─────────────────────────────
  #for baseline in "${baseline_list[@]}"; do
  #  echo "===================="
  #  echo "[Baseline] Exporting FP${baseline} for ${model}"
  #  python export_and_quant.py \
  #    --model_name "${model}" \
  #    --wq_bits "${baseline}"    \
      # --wq_datatype 없이 baseline 처리
  #done

  # ─── Quantization ─────────────────────────
  for wq_bit in "${wq_bit_list[@]}"; do
    if [[ "$wq_bit" -eq 32 || "$wq_bit" -eq 16 ]]; then
      continue
    fi

    # datatype list
    case $wq_bit in
      3)  datatype_list=(int3 int3_asym fp3)      ;;
      4)  datatype_list=(int4 int4_asym fp4)  ;;
      5)  datatype_list=(fp5_e2m2 fp5_e3m1)      ;; # (int5 int5_asym fp5_e2m2 fp5_e3m1) 
      6)  datatype_list=(fp6_e2m3 fp6_e3m2) ;;      # (int6 int6_asym fp6_e2m3 fp6_e3m2)
      7)  datatype_list=(int7 int7_asym)      ;;
      8)  datatype_list=(int8 int8_asym fp8_e2m5 fp8_e3m4 fp8_e4m3 fp8_e5m2)      ;; # (int8 int8_asym fp8_e2m5 fp8_e3m4 fp8_e4m3 fp8_e5m2)
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
