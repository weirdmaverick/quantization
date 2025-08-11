#!/usr/bin/env python3
import os
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt

# quant_error_single_weight.py 
from quant_error_single_weight import (
    load_initializers,
    proto_to_array,
    parse_quant_config,
    compute_errors,
)

TARGET_INIT = "model.decoder.layers.0.self_attn.k_proj.weight"

def main(baseline_dir, quant_dirs, output_csv):
    # 1) baseline load
    base_path = os.path.join(baseline_dir, "model.onnx")
    base_inits = load_initializers(base_path)
    if TARGET_INIT not in base_inits:
        raise ValueError(f"{TARGET_INIT} not found in baseline.")
    arr_base = proto_to_array(base_inits[TARGET_INIT])

    rows = []
    per_vecs = {}

    for qdir in quant_dirs:
        qpath = os.path.join(qdir, "model.onnx")
        grouping, dtype, bits, gs = parse_quant_config(qpath)

        quant_inits = load_initializers(qpath)
        if TARGET_INIT not in quant_inits:
            print(f"[warn] skip {qdir}, no {TARGET_INIT}")
            continue
        arr_quant = proto_to_array(quant_inits[TARGET_INIT])

        # compute stats
        stats = compute_errors(arr_base, arr_quant, grouping, gs)

        # 채널/그룹 수 계산
        if arr_quant.ndim == 2:
            K, C = arr_quant.shape
        else:
            # fallback for weird shapes
            K = arr_quant.shape[0]
            C = 1
        num_channels = K
        groups_per_channel = C // gs if (grouping=="per-group" and gs) else ""

        # CSV row
        row = {
            "datatype":           dtype,
            "granularity":        grouping,
            "bits":               bits or "",
            "num_channels":       num_channels,
            "groups_per_channel": groups_per_channel,
            "mse_tensor":         stats["mse_tensor"],
            "mse_rep":            stats.get("mse_rep", ""),
            "mse_p99":            stats.get("mse_p99", ""),
        }
        rows.append(row)

        # per‐vec 모아두기
        if "per_vec" in stats:
            label = f"{dtype} ({grouping})"
            per_vecs[label] = np.array(stats["per_vec"])

    # 2) CSV write
    if rows:
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"[done] report → {output_csv}")

    # 3) 히스토그램 그리기
    if per_vecs:
        plt.figure(figsize=(8,5))
        bins = 50
        for label, vec in per_vecs.items():
            plt.hist(
                vec,
                bins=bins,
                histtype='step',
                linewidth=1.5,
                label=f"{label}  (channels={num_channels}"
                      + (f", #group(group size = 128)={groups_per_channel})" if groups_per_channel else ")")
            )
        plt.xlabel("Channel-(or-group) MSE")
        plt.ylabel("Count")
        plt.title(f"Per-channel/group MSE for '{TARGET_INIT}'")
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("No per-channel/group data to plot.")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Per-channel and per-group MSE report + histogram")
    p.add_argument("--baseline_dir", required=True,
                   help="Directory of FP16 baseline (contains model.onnx)")
    p.add_argument("--quant_dirs", nargs="+", required=True,
                   help="List of quant directories (each containing model.onnx)")
    p.add_argument("--output_csv", default="single_weight_error.csv",
                   help="Path to save CSV report")
    args = p.parse_args()
    main(args.baseline_dir, args.quant_dirs, args.output_csv)
    
    
'''
./plot.py   --baseline_dir ../onnx_graph/facebook__opt-1.3b/fp16_baseline  \
    --quant_dirs     ../onnx_graph/facebook__opt-1.3b/per-channel/int8_asym/w_8_gs_none \
    ../onnx_graph/facebook__opt-1.3b/per-channel/fp8_e3m4/w_8_gs_none \
    ../onnx_graph/facebook__opt-1.3b/per-channel/fp8_e4m3/w_8_gs_none \
    ../onnx_graph/facebook__opt-1.3b/per-channel/int8/w_8_gs_none \
    --output_csv channel_mse_report.csv

./plot.py   --baseline_dir ../onnx_graph/facebook__opt-1.3b/fp16_baseline  \
    --quant_dirs     ../onnx_graph/facebook__opt-1.3b/per-group/int8_asym/w_8_gs_128 \
    ../onnx_graph/facebook__opt-1.3b/per-group/fp8_e3m4/w_8_gs_128 \
    ../onnx_graph/facebook__opt-1.3b/per-group/fp8_e4m3/w_8_gs_128 \
    ../onnx_graph/facebook__opt-1.3b/per-group/int8/w_8_gs_128 \
    --output_csv group_mse_report.csv
'''
