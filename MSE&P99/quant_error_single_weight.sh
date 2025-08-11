#!/bin/bash
set -euo pipefail

# 1) Baseline directory (fp16)
BASELINE="../onnx_graph/facebook__opt-1.3b/fp16_baseline"

# 2) Quant directory
MODEL_DIR="../onnx_graph/facebook__opt-1.3b"

# 3) output CSV
OUT="weight_errors_.csv"

# 4) per-channel, per-group 아래의 model.onnx 파일이 있는 디렉토리만 추출
mapfile -t QUANT_DIRS < <(
  find "${MODEL_DIR}" -type f \
    \( -path "*/per-channel/*/model.onnx" -o -path "*/per-group/*/model.onnx" \) \
    -exec dirname {} \; | sort -u
)

if [ ${#QUANT_DIRS[@]} -eq 0 ]; then
  echo "[Error] cannot find quant directory for analysis." >&2
  exit 1
fi

echo "Found quant dirs:"
printf '  %s\n' "${QUANT_DIRS[@]}"

# 5) quant_error.py 호출
python quant_error.py \
  --baseline_dir "${BASELINE}" \
  --quant_dirs "${QUANT_DIRS[@]}" \
  --output_csv "${OUT}"

echo "[Done] quant error for all datatypes → ${OUT}"
