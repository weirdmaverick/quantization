import os, gc
import onnx
import numpy as np
import matplotlib.pyplot as plt
from onnx import TensorProto

# —————————————— 설정 ——————————————
BASELINE_ONNX = "onnx_graph/facebook__opt-1.3b/fp16_baseline/model.onnx"
DTYPE_MAP = {
    TensorProto.FLOAT:   np.float32,
    TensorProto.FLOAT16: np.float16,
    TensorProto.DOUBLE:  np.float64,
    TensorProto.UINT8:   np.uint8,
    TensorProto.INT8:    np.int8,
    TensorProto.INT32:   np.int32,
    TensorProto.INT64:   np.int64,
}
BINS       = 200
CHUNK_SIZE = 500_000

def load_model_and_inits(path):
    model = onnx.load(path)
    inits = {init.name: init for init in model.graph.initializer}
    return model, inits

def find_compute_weights(model, inits):
    init_map = set(inits)
    nodes = [n for n in model.graph.node if n.op_type in ("MatMul","Gemm")]
    out2node = {out: n for n in model.graph.node for out in n.output}
    W = set()
    for n in nodes:
        if len(n.input)<2: continue
        inp = n.input[1]
        if inp in init_map:
            W.add(inp)
        else:
            tn = out2node.get(inp)
            if tn and tn.op_type=="Transpose" and tn.input[0] in init_map:
                W.add(tn.input[0])
    return sorted(W)

def main():
    # 1) load
    model, inits = load_model_and_inits(BASELINE_ONNX)
    compute_ws   = find_compute_weights(model, inits)

    # 2) 각 weight 수(원소) 구하고 offsets 계산
    lengths = [int(np.prod(inits[name].dims)) for name in compute_ws]
    offsets = np.concatenate(([0], np.cumsum(lengths)[:-1]))

    # ──────────── 두-패스 히스토그램 계산 ────────────
    # 2.1) global min/max
    gmin, gmax = np.inf, -np.inf
    for name in compute_ws:
        init     = inits[name]
        dtype    = DTYPE_MAP[init.data_type]
        raw_buf  = init.raw_data
        N        = int(np.prod(init.dims))
        step     = CHUNK_SIZE
        itemsize = np.dtype(dtype).itemsize

        pos = 0
        while pos < N:
            L = min(step, N-pos)
            arr = (
                np.frombuffer(raw_buf,
                             dtype=dtype,
                             count=L,
                             offset=pos*itemsize)
                .astype(np.float32)
            )
            gmin, gmax = min(gmin, arr.min()), max(gmax, arr.max())
            pos += L
            del arr

    # 2.2) bin_edges & counts
    bin_edges = np.linspace(gmin, gmax, BINS+1)
    counts    = np.zeros(BINS, dtype=np.int64)

    for name in compute_ws:
        init     = inits[name]
        dtype    = DTYPE_MAP[init.data_type]
        raw_buf  = init.raw_data
        N        = int(np.prod(init.dims))
        step     = CHUNK_SIZE
        itemsize = np.dtype(dtype).itemsize

        pos = 0
        while pos < N:
            L = min(step, N-pos)
            arr = (
                np.frombuffer(raw_buf,
                             dtype=dtype,
                             count=L,
                             offset=pos*itemsize)
                .astype(np.float32)
            )
            ct, _ = np.histogram(arr, bins=bin_edges)
            counts += ct
            pos += L
            del arr

    max_count = counts.max()

    # ──────────── scatter plot ────────────
    plt.figure(figsize=(12,4))
    for idx, name in enumerate(compute_ws):
        init     = inits[name]
        dtype    = DTYPE_MAP[init.data_type]
        raw_buf  = init.raw_data
        N        = int(np.prod(init.dims))
        step     = CHUNK_SIZE
        itemsize = np.dtype(dtype).itemsize
        

        print(f"[{idx+1}/{len(compute_ws)}] Plotting {name} ({N} elems)")

        pos = 0
        while pos < N:
            L = min(step, N-pos)
            arr = (
                np.frombuffer(raw_buf,
                             dtype=dtype,
                             count=L,
                             offset=pos*itemsize)
                .astype(np.float32)
            )

            # density→alpha
            raw_idx    = np.clip(np.digitize(arr, bin_edges)-1, 0, BINS-1)
            dens       = counts[raw_idx] / max_count
            alpha_vals = dens * 0.9 + 0.1

            cols       = np.zeros((L,4), dtype=np.float32)
            cols[:,2]  = 1.0
            cols[:,3]  = alpha_vals

            x = np.full(L, idx, dtype=np.int32)

            plt.scatter(x, arr,
                        s=1,
                        c=cols,
                        marker='.',
                        rasterized=True)

            pos += L
            del arr, cols, alpha_vals, dens, raw_idx, x
            gc.collect()

    plt.xlabel("Weight Index")
    plt.ylabel("Value")
    plt.title("Weights of MatMul/Gemm initializers (fp16 baseline)")
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    main()



""" import os
import gc                                  # ← 1) gc 모듈 import
import onnx
import numpy as np
import matplotlib.pyplot as plt
from onnx import TensorProto

# 1) fp16 baseline onnx 경로
BASELINE_ONNX = "onnx_graph/facebook__opt-1.3b/fp16_baseline/model.onnx"

# 2) ONNX → NumPy dtype 매핑
DTYPE_MAP = {
    TensorProto.FLOAT:    np.float32,
    TensorProto.FLOAT16:  np.float16,
    TensorProto.DOUBLE:   np.float64,
    TensorProto.UINT8:    np.uint8,
    TensorProto.INT8:     np.int8,
    TensorProto.INT32:    np.int32,
    TensorProto.INT64:    np.int64,
}

# histogram bin counts, chunk size
BINS = 200
CHUNK_SIZE = 500_000

def load_model_and_initializers(path):
    model = onnx.load(path)
    inits = {init.name: init for init in model.graph.initializer}
    return model, inits

def proto_to_array(init):
    np_dtype = DTYPE_MAP[init.data_type]
    arr = np.frombuffer(init.raw_data, dtype=np_dtype).reshape(init.dims)
    return arr.astype(np.float32)

def find_compute_weights(model):
    init_map = {init.name: init for init in model.graph.initializer}
    nodes = [n for n in model.graph.node if n.op_type in ("MatMul","Gemm")]
    out2node = {out: n for n in model.graph.node for out in n.output}
    weights = set()
    for n in nodes:
        if len(n.input)<2: continue
        inp = n.input[1]
        if inp in init_map:
            weights.add(inp)
        else:
            tn = out2node.get(inp)
            if tn and tn.op_type=="Transpose":
                w0 = tn.input[0]
                if w0 in init_map:
                    weights.add(w0)
    return sorted(weights)

def main():
    print("[Info] Loading ONNX model and initializers...")
    model, inits = load_model_and_initializers(BASELINE_ONNX)
    compute_ws = find_compute_weights(model)
    total = len(compute_ws)
    print(f"[Info] Found {total} MatMul/Gemm weights to plot.\n")

    # ──────────────── 두-패스 히스토그램 계산 ────────────────
    print("[Info] Scanning global min/max for bins...")
    global_min, global_max = np.inf, -np.inf
    for name in compute_ws:
        arr = proto_to_array(inits[name]).ravel()
        global_min = min(global_min, arr.min())
        global_max = max(global_max, arr.max())
        del arr

    bin_edges = np.linspace(global_min, global_max, BINS+1)
    counts = np.zeros(BINS, dtype=np.int64)
    print("[Info] Accumulating histogram counts...")
    for name in compute_ws:
        arr = proto_to_array(inits[name]).ravel()
        ct, _ = np.histogram(arr, bins=bin_edges)
        counts += ct
        del arr
    max_count = counts.max()

    # ──────────────── plot ────────────────
    plt.figure(figsize=(12, 4))
    plt.ion()

    for idx, name in enumerate(compute_ws):
        init      = inits[name]
        dtype     = DTYPE_MAP[init.data_type]
        itemsize  = np.dtype(dtype).itemsize
        raw_buf   = init.raw_data
        N         = int(np.prod(init.dims))
        pos = 0

        print(f"[{idx+1}/{total}] Plotting '{name}'  elements={N}", flush=True)

        while pos < N:
            length = min(CHUNK_SIZE, N - pos)
            arr_chunk = (np
                .frombuffer(raw_buf,
                            dtype=dtype,
                            count=length,
                            offset=pos * itemsize)
                .astype(np.float32)
            )

            # density 기반 alpha 계산
            raw_idx    = np.digitize(arr_chunk, bin_edges) - 1
            bin_idx    = np.clip(raw_idx, 0, BINS-1)
            dens       = counts[bin_idx] / max_count
            alpha_vals = dens * 0.9 + 0.1

            # 1) x-coordinates
            x = np.full(length, idx, dtype=np.int32)

            # 2) RGBA array: all-blue with per-point alpha
            colors = np.zeros((length, 4), dtype=np.float32)
            colors[:,   2] = 1.0         # blue channel
            colors[:,   3] = alpha_vals  # alpha channel

            # 3) scatter with per-point RGBA, rasterized
            plt.scatter(x, arr_chunk,
                        s=1,
                        c=colors,
                        marker='.',
                        rasterized=True)
            pos += length
            # ← 3) gc 강제 수집
            del arr_chunk, raw_idx, bin_idx, dens, alpha_vals, x
            gc.collect()


    plt.xlabel("Weight Tensor Index")
    plt.ylabel("Value")
    plt.title("…")
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    main() """
