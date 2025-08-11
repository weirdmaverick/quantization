""" #!/usr/bin/env python3
import argparse
import numpy as np
from single_weight_quant_error import load_weight

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-b","--baseline_path", required=True,
                   help="FP16 baseline ONNX file path")
    p.add_argument("-q","--dequant", action="append", nargs=2, metavar=('LABEL','PATH'),
                   required=True,
                   help="Quant format label and ONNX path")
    p.add_argument("-t","--tensor_name", required=True,
                   help="Initializer tensor name")
    p.add_argument("-th","--outlier_threshold", type=float, required=True,
                   help="Error threshold for outliers (float)")
    return p.parse_args()

def compress_ranges(idxs):
    
    if not idxs:
        return ""
    ranges = []
    start = prev = idxs[0]
    for x in idxs[1:]:
        if x == prev + 1:
            prev = x
        else:
            # start==prev면 단일값, else 범위
            ranges.append(f"{start}-{prev}" if start != prev else str(start))
            start = prev = x
    # 마지막 구간 추가
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ", ".join(ranges)

def main():
    args = parse_args()

    # 1) baseline load
    w_fp16 = load_weight(args.baseline_path, args.tensor_name).flatten()

    # 2) load dequant & compute errors
    labels, errors = [], []
    for label, path in args.dequant:
        w_deq = load_weight(path, args.tensor_name).flatten()
        errors.append(np.abs(w_fp16 - w_deq))
        labels.append(label)
    errors = np.stack(errors, axis=0)  # shape: (F, N)

    # 3) outlier detection
    min_err = errors.min(axis=0)
    mask_out = min_err > args.outlier_threshold
    outliers = np.where(mask_out)[0].tolist()
    outlier_ranges = compress_ranges(sorted(outliers))

    # 4) valid indices
    valid_idx = np.where(~mask_out)[0]

    # 5) choose best format per-element
    best_fmt = errors[:, valid_idx].argmin(axis=0)
    best_indices = { lbl: [] for lbl in labels }
    for pos, fmt_i in enumerate(best_fmt):
        lbl = labels[fmt_i]
        elem = valid_idx[pos]
        best_indices[lbl].append(int(elem))

    # 6) print results as ranges
    print(f"Outlier indices (min error > {args.outlier_threshold}):")
    print(outlier_ranges or "None")
    print()
    for lbl in labels:
        rng = compress_ranges(sorted(best_indices[lbl]))
        print(f"{lbl} best index ranges ({len(best_indices[lbl])} elements):")
        print(rng or "None")
        print()

if __name__ == "__main__":
    main()
 """
import onnx
import numpy as np
import matplotlib.pyplot as plt
from single_weight_quant_error import load_weight

# ───────────────────────────────────────────────────────────────
# 1) 설정
# ───────────────────────────────────────────────────────────────
baseline_path = "../onnx_graph/facebook__opt-1.3b/fp16_baseline/model.onnx"
formats = {
    "int4_asym": "../onnx_graph/facebook__opt-1.3b/per-group/int4_asym/w_4_gs_128/model.onnx",
    "fp4"      : "../onnx_graph/facebook__opt-1.3b/per-group/fp4/w_4_gs_128/model.onnx"
}
tensor_name = "model.decoder.layers.0.self_attn.k_proj.weight"
threshold   = 0.02

# ───────────────────────────────────────────────────────────────
# 2) 에러 계산 (절댓값 기준)
# ───────────────────────────────────────────────────────────────
w_fp16 = load_weight(baseline_path, tensor_name).flatten()
errors = {}
for lbl, path in formats.items():
    w_q = load_weight(path, tensor_name).flatten()
    
    errors[lbl] = np.abs(w_fp16 - w_q)

all_err = np.stack(list(errors.values()), axis=0)
min_err = all_err.min(axis=0)

# ───────────────────────────────────────────────────────────────
# 3) 인덱스 분류
# ───────────────────────────────────────────────────────────────
indices = np.arange(min_err.size)

#  outlier : min_err > threshold
mask_out = min_err > threshold
outlier_idx = indices[mask_out]

# valid : min_err ≤ threshold
mask_valid = ~mask_out

# 각 format 별 best 인덱스 (절댓값 기준)
best_indices = {}
for i, lbl in enumerate(formats):
    mask_best = (all_err[i] == min_err) & mask_valid
    best_indices[lbl] = np.where(mask_best)[0]

# ───────────────────────────────────────────────────────────────
# 4) 연속 구간으로 묶어주는 헬퍼
# ───────────────────────────────────────────────────────────────
def to_ranges(idx_array):
    idx = np.sort(idx_array)
    if idx.size == 0:
        return []
    ranges = []
    start = prev = idx[0]
    for x in idx[1:]:
        if x == prev + 1:
            prev = x
        else:
            ranges.append((start, prev - start + 1))
            start = prev = x
    ranges.append((start, prev - start + 1))
    return ranges

ranges_out = to_ranges(outlier_idx)
ranges_int4 = to_ranges(best_indices["int4_asym"])
ranges_fp4 = to_ranges(best_indices["fp4"])

# ───────────────────────────────────────────────────────────────
# 5) 시각화
# ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 3))

#  outlier   (y=2)
ax.broken_barh(ranges_out, (2, 0.8), facecolors='lightgray')
#  int4_asym (y=1)
ax.broken_barh(ranges_int4, (1, 0.8), facecolors='C1')
#  fp4       (y=0)
ax.broken_barh(ranges_fp4, (0, 0.8), facecolors='C0')

ax.set_ylim(-0.5, 3)
ax.set_yticks([0.4, 1.4, 2.4])
ax.set_yticklabels(["fp4", "int4_asym", "OUTLIER"])
ax.set_xlabel("Element index")
ax.set_title("Per-element Best Format Ranges + OUTLIER ")
ax.grid(True, axis='x')
plt.tight_layout()
plt.show()
