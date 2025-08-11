import os
import argparse
import onnx
import numpy as np
import csv
from onnx import TensorProto

# Script: Compute quant error for all MatMul/Gemm weights across entire initializer set

import argparse
import onnx
import numpy as np
import csv
from onnx import TensorProto

# Script: Compute quantization error for all MatMul/Gemm weights

# ONNX -> NumPy dtype mapping
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
    return {init.name: init for init in model.graph.initializer}, model

# raw bytes in TensorProto object -> numpy array
def proto_to_array(init: onnx.TensorProto):
    dtype = DTYPE_MAP[init.data_type]
    arr = np.frombuffer(init.raw_data, dtype=dtype).reshape(init.dims)
    return arr.astype(np.float32)

# model path parsing
def parse_quant_config(model_path: str):
    parts = model_path.split(os.sep)
    for i, p in enumerate(parts):
        if p in ("per-channel", "per-group"):
            grouping = p
            dtype    = parts[i+1]
            _, bits, _, gs = parts[i+2].split("_")
            gs = None if gs == "none" else int(gs)
            return grouping, dtype, int(bits), gs
    # baseline folder
    base = os.path.basename(os.path.dirname(model_path))
    return "baseline", base, None, None


def compute_mse(a: np.ndarray, b: np.ndarray):
    a32, b32 = a.astype(np.float32), b.astype(np.float32)
    diff = a32 - b32
    return float(np.mean(diff * diff))


def compute_errors(a: np.ndarray, b: np.ndarray, grouping, gs):
    # 1) per-tensor MSE
    mse_tensor = compute_mse(a, b)

    # 2) per-channel or per-group vector MSE (2D only)
    per_vec = None
    if grouping == "per-channel" and a.ndim == 2:
        _, C = a.shape
        per_vec = [compute_mse(a[:, c], b[:, c]) for c in range(C)]
    elif grouping == "per-group" and a.ndim == 2:
        _, C = a.shape
        assert gs and C % gs == 0, "group size mismatch"
        G = C // gs
        per_vec = []
        for g in range(G):
            slice_a = a[:, g*gs:(g+1)*gs]
            slice_b = b[:, g*gs:(g+1)*gs]
            per_vec.append(compute_mse(slice_a, slice_b))

    result = {"mse_tensor": mse_tensor}
    if per_vec is not None:
        vec = np.array(per_vec)
        result.update({
            "mse_rep": float(vec.mean()),
            "mse_p99": float(np.percentile(vec, 99)),
        })
    return result

# Sort weights for GeMM, MatMul 
def find_compute_weights(model):
    # Identify initializers used in MatMul/Gemm
    initializer_map = {init.name: init for init in model.graph.initializer}
    matmul_nodes = [n for n in model.graph.node if n.op_type in ("MatMul", "Gemm")]
    # transpose output -> node lookup table
    output_to_node = {}
    for node in model.graph.node:
        for out_name in node.output:
            output_to_node[out_name] = node

    compute_weights = set()
    for node in matmul_nodes:
        if len(node.input) < 2:
            continue
        inp = node.input[1] # input[0] : A, GeMM node's input B
        # Case A) GeMM: second input is used as a weight initializer
        if inp in initializer_map:
            compute_weights.add(inp)
        # Case B) MatMul: input B is used as "Transpose" node's output,
        #                 "Transpose" nodes gets weight initializer as an input and returns that output
        else:
            tn = output_to_node.get(inp)
            if tn and tn.op_type == "Transpose":
                w_name = tn.input[0]
                if w_name in initializer_map:
                    compute_weights.add(w_name)
    return compute_weights


def main(baseline_dir, quant_dir, output_csv):
    # Load baseline and quant models + initializers
    base_fp = os.path.join(baseline_dir, "model.onnx")
    quant_fp = os.path.join(quant_dir,    "model.onnx")

    base_inits,     base_model  = load_initializers(base_fp)
    quant_inits,    quant_model = load_initializers(quant_fp)

    # Compute set of weight names involved in MatMul/Gemm
    compute_weights = find_compute_weights(base_model)

    grouping, dtype, bits, gs = parse_quant_config(quant_fp)
    rows = []

    for name, init_base in base_inits.items():
        # filter only compute_weights present in both baseline and quant
        if name not in compute_weights or name not in quant_inits:
            continue

        arr_base  = proto_to_array(init_base)
        arr_quant = proto_to_array(quant_inits[name])
        stats = compute_errors(arr_base, arr_quant, grouping, gs)

        row = {
            "initializer": name,
            "datatype":    dtype,
            "granularity": grouping,
            "bits":        bits or "",
            "mse_tensor":  stats["mse_tensor"],
            "mse_rep":     stats.get("mse_rep", ""),
            "mse_p99":     stats.get("mse_p99", ""),
        }
        rows.append(row)

    if not rows:
        print("[Error] No MatMul/Gemm weights found.")
        return

    # Write CSV
    keys = list(rows[0].keys())
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Done] report saved -> {output_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--baseline_dir", required=True)
    p.add_argument("--quant_dir",    required=True)
    p.add_argument("--output_csv",   default="all_weights_error_filtered.csv")
    args = p.parse_args()
    main(args.baseline_dir, args.quant_dir, args.output_csv)
