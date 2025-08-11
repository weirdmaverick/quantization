import os
import argparse
import onnx
import numpy as np
import csv
import matplotlib.pyplot as plt
from onnx import TensorProto
import matplotlib.pyplot as plt
# --------------------
# initializer 
TARGET_INIT = "model.decoder.layers.0.self_attn.k_proj.weight"

# ONNX → NumPy dtype mapping
DTYPE_MAP = {
    TensorProto.FLOAT:   np.float32,
    TensorProto.FLOAT16: np.float16,
    TensorProto.DOUBLE:  np.float64,
    TensorProto.INT8:    np.int8,
    TensorProto.UINT8:   np.uint8,
    TensorProto.INT32:   np.int32,
    TensorProto.INT64:   np.int64,
}

def load_initializers(model_path: str):
    model = onnx.load(model_path)
    return {init.name: init for init in model.graph.initializer}

def proto_to_array(init: onnx.TensorProto):
    dtype = DTYPE_MAP[init.data_type]
    arr = np.frombuffer(init.raw_data, dtype=dtype)
    return arr.reshape(init.dims).astype(np.float32)

def parse_quant_config(model_path: str):
    parts = model_path.split(os.sep)
    for i, p in enumerate(parts):
        if p in ("per-channel", "per-group"):
            grouping = p
            dtype    = parts[i+1]
            _, bits, _, gs = parts[i+2].split("_")
            gs = None if gs=="none" else int(gs)
            return grouping, dtype, int(bits), gs
    # baseline folder
    base = os.path.basename(os.path.dirname(model_path))
    return "baseline", base, None, None

def compute_mse(a: np.ndarray, b: np.ndarray):
    diff = a - b
    return float(np.mean(diff*diff))

def compute_errors(a: np.ndarray, b: np.ndarray, grouping, gs):
    # 1) per-tensor (global) MSE
    mse_tensor = compute_mse(a, b)

    # 2) per-channel or per-group vector of MSEs
    per_vec = None
    if grouping == "per-channel" and a.ndim == 2:
        # Row as “# of channels” [K,C]
        K, _ = a.shape
        per_vec = [ compute_mse(a[k,:], b[k,:]) for k in range(K) ]

    elif grouping == "per-group" and a.ndim == 2:
        K, C = a.shape          
        assert gs and C % gs == 0, "group size mismatch"
        G = C // gs     # [K,g,gs(=128)]
        per_vec=[]
        for k in range(K):
            for g  in range(G):
                start, end = g*gs, (g+1)*gs
                per_vec.append(
                    compute_mse(a[k, start:end], b[k, start:end])
                )

    # 3) representative (mean of per_vec), plus percentiles
    result = {"mse_tensor": mse_tensor}
    if per_vec is not None:
        vec = np.array(per_vec)
        result["per_vec"]   = vec.tolist()
        result["mse_rep"]   = float(vec.mean())
        result["mse_p99"]   = float(np.percentile(vec, 99))

    plt.hist(np.abs(per_vec), bins = 200)
    plt.title(f"{grouping} MSE distribution")
    return result

def main(baseline_dir, quant_dirs, output_csv):
    # load baseline
    base_path  = os.path.join(baseline_dir, "model.onnx")
    base_inits = load_initializers(base_path)
    if TARGET_INIT not in base_inits:
        raise ValueError(f"{TARGET_INIT} not in baseline.")
    arr_base = proto_to_array(base_inits[TARGET_INIT])

    rows = []
    for qdir in quant_dirs:
        qpath   = os.path.join(qdir, "model.onnx")
        grouping, dtype, bits, gs = parse_quant_config(qpath)

        print(f"\n[DEBUG] {dtype} @ {grouping}, bits={bits}, gs={gs}")

        quant_inits = load_initializers(qpath)
        if TARGET_INIT not in quant_inits:
            print(f"[warn] skip {qdir}")
            continue
        arr_quant = proto_to_array(quant_inits[TARGET_INIT])

        stats = compute_errors(arr_base, arr_quant, grouping, gs)

        row = {
            "datatype":    dtype,
            "granularity": grouping,
            "bits":        bits or "",
            "mse_tensor":  stats["mse_tensor"],
            # for per-channel/group
            "mse_rep":     stats.get("mse_rep", ""),
            "mse_p99":     stats.get("mse_p99", ""),
        }
        rows.append(row)
        if per_vecs:
            plt.figure(figsize=(8,5))
            bins = 50
            for label, vec in per_vecs.items():
                plt.hist(vec, bins=bins, histtype='step', density=False, 
                        label=label, linewidth=1.5)
            plt.xlabel("Channel-wise MSE")
            plt.ylabel("Count")
            plt.title("Per-channel MSE Distribution")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    # write CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[done] report → {output_csv}")
    
    per_vecs={}
    
    
    

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--baseline_dir", required=True)
    p.add_argument("--quant_dirs", nargs="+", required=True)
    p.add_argument("--output_csv", default="single_weight_error.csv")
    args = p.parse_args()
    main(args.baseline_dir, args.quant_dirs, args.output_csv)
