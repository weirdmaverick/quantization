#!/usr/bin/env python3
import argparse
import onnx
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute and plot bucketed quantization error between FP16 and another quant format")
    p.add_argument("--baseline-path",   "-b", required=True,
                   help="FP16 baseline ONNX file path")
    p.add_argument("--quant-path",      "-q", required=True,
                   help="Quantized ONNX file path (INT4/INT8/FP4/etc.)")
    p.add_argument("--tensor-name",     "-t", required=True,
                   help="Initializer tensor name (e.g. model.decoder.layers.0.self_attn.k_proj.weight)")
    p.add_argument("--step-size",       "-s", type=float, default=0.0001,
                   help="Bucket step size (default: 0.0001)")
    p.add_argument("--max-value",       "-m", type=float, default=0.0007,
                   help="Max error value to bucket (values above clipped to max-step, default: 0.0007)")
    p.add_argument("--title",           "-T", default="Quantized Error Bucket Plot",
                   help="Plot title")
    return p.parse_args()

def load_weight(onnx_path: str, tensor_name: str) -> np.ndarray:
    model = onnx.load(onnx_path)
    for init in model.graph.initializer:
        if init.name == tensor_name:
            return onnx.numpy_helper.to_array(init)
    raise KeyError(f"Initializer '{tensor_name}' not found in {onnx_path}")

def main():
    args = parse_args()

    # 1) load weights
    w_fp16 = load_weight(args.baseline_path, args.tensor_name)
    w_q    = load_weight(args.quant_path,    args.tensor_name)

    # 2) compute error
    err = (w_fp16 - w_q).flatten()
    abs_err = np.abs(err)

    # 3) build buckets
    step   = args.step_size
    # bins: [step, 2*step, ..., max_value+step]
    bins   = np.arange(step, args.max_value + step, step)
    # bucket values: [0, step, 2*step, ..., (len(bins)-1)*step]
    values = np.arange(len(bins)) * step

    # 4) digitize → indices into values
    idx = np.digitize(abs_err, bins, right=True)
    idx = np.minimum(idx, len(values)-1)
    quant_err = values[idx]

    # 5) plot
    indices = np.arange(quant_err.shape[0], dtype=int)
    plt.figure(figsize=(12,4))
    plt.step(indices, quant_err, where='mid', lw=0.5)
    plt.xlabel('Element index')
    plt.ylabel(f'Error (step={step})')
    plt.title(args.title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
