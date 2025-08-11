# onnx_analysis.py
import onnx, numpy as np, os
import torch
from onnx import TensorProto
from quant_utils.quant_weight import quant_int, quant_int_asym, quant_datatype

def analyze_quant_initializers(model_path: str):
    print(f"\n=== Analyzing initializers in {model_path} ===\n")
    model = onnx.load(model_path)
    initializer_map = {init.name: init for init in model.graph.initializer}

    # 1) MatMul/Gemm node
    matmul_nodes = [n for n in model.graph.node if n.op_type in ("MatMul", "Gemm")]

    # transpose output -> node lookup table
    output_to_node = {}
    for n in model.graph.node:
        for out_name in n.output:
            output_to_node[out_name] = n 
    
    
    # 2) weight initializer tracking
    compute_weights = set()
    for n in matmul_nodes:
        if len(n.input) < 2:
            continue
        inp = n.input[1]
        
        # case A) GeMM : input B is used as a weight initializer
        if inp in initializer_map:
            compute_weights.add(inp)
            continue
        
        # case B) MatMul: input B is used as a "Transpose" node's output, 
        #                 "Transpose" node gets weight initializer as an input and returns that output
        tn = output_to_node.get(inp)
        if tn is not None and tn.op_type == "Transpose":
            # Check if the "Transpose" node's input is weight initializer
            w_name = tn.input[0]
            if w_name in initializer_map:
                compute_weights.add(w_name)
                
    # 3) weight / bias / weight_q 
    weight_inits = [init for init in model.graph.initializer
                    if any(k in init.name.lower() for k in ("weight", "bias", "weight_q", "weight_scale", "weight_zp"))]

    # 4) grouping·bits·groupsize parsing
    parts, grouping, dtype_dir, bits, gs = model_path.split(os.sep), None, None, None, None
    for idx, p in enumerate(parts):
        if p in ("per-group", "per-channel"):
            grouping = p
            dtype_dir = parts[idx + 1]
            _, bit_s, _, gs_s = parts[idx + 2].split("_")
            bits = int(bit_s)
            gs = None if gs_s == "none" else int(gs_s)
            break

    # 5) onnx dtype → numpy dtype
    dtype_map = {TensorProto.FLOAT  : np.float32,
                 TensorProto.FLOAT16: np.float16,
                 TensorProto.DOUBLE : np.float64,
                 TensorProto.UINT8  : np.uint8,
                 TensorProto.INT8   : np.int8,
                 TensorProto.INT32  : np.int32,
                 TensorProto.INT64  : np.int64}

    # 6) initializer loop
    for init in weight_inits:
        name, is_compute, dims      = init.name, init.name in compute_weights, list(init.dims)
        np_dtype, raw               = dtype_map.get(init.data_type), init.raw_data

        print(f"--- Initializer: {name} ---")
        print(f"Used in MatMul/Gemm? {'YES' if is_compute else 'no'}")
        print(f"Shape         : {dims}")
        print(f"DataType enum : {init.data_type}")

        if not raw or np_dtype is None:
            print("!! Cannot parse raw_data.\n"); continue

        arr       = np.frombuffer(raw, dtype=np_dtype).reshape(dims)
        print(f"Raw elements  : {arr.size} (expected {int(np.prod(dims))})")

        # ─────────────────────────────────────────────────────────────
        #  (A) quantized tensor (weight_q) 
        # ─────────────────────────────────────────────────────────────      
        if name.endswith('.weight_q'):
            # arr may be either [K, C] (per-channel) or [K, G, S] (per-group)
            if arr.ndim == 3:
                K, G, S = arr.shape
                for ch in range(min(K, 4)): # channel 0~3 sample
                    for g in range(min(G, 5)): # group 0~4 sample            
                        sample = arr[ch, g, :32].tolist()
                        print(f"Ch{ch:4d} Gr{g:4d}: q_vals={sample}…")
                    print()                    
            else:
                K, C = arr.shape
                for ch in range(min(K, 16)):
                    sample = arr[ch, :32].tolist()
                    print(f"Ch{ch:4d}: q_vals={sample}…")
                    if ch == 16:
                        break

            print(f"Min / Max      : {arr.min():.0f} / {arr.max():.0f}\n")

        if name.endswith('weight_scale') or name.endswith('weight_zp'):
            if arr.ndim == 1:   # per-channel
                K = arr.shape[0]
                for ch in range(min(K, 16)):
                    print(f"Ch{ch:4d}: {arr[ch]:.6f}")
            else:                   # per-group
                K, G = arr.shape
                for ch in range(min(K, 4)):
                    print()
                    for g in range(min(G, 5)):
                        print(f"Ch{ch:4d} Gr{g:4d}: {arr[ch, g]:.6f}")
                print()
                    
            print(f"Min / Max      : {np.min(arr):.6f} / {np.max(arr):.6f}\n")
        # ─────────────────────────────────────────────────────────────
        #  (B) FP16 compute weight 
        # ─────────────────────────────────────────────────────────────
        if is_compute:
            print(f"Vals(sample)  : {arr.flatten()[:32].tolist()} …")
            print(f"Min / Max      : {arr.min():.6f} / {arr.max():.6f}")

        # ─────────────────────────────────────────────────────────────
        #  (C) Demo-quantization based on FP16 baseline 
        # ─────────────────────────────────────────────────────────────
        if 'fp16_baseline' in model_path and is_compute and len(dims) == 2 and bits is not None:
            W = torch.from_numpy(arr).to(torch.float16)
            W_int = quant_int(W, wq_bits=bits, group_size=gs)
            print("\n[Sym INT] sample:", W_int.view(-1)[:32].tolist())
            print()
            W_int_asym = quant_int_asym(W, wq_bits=bits, group_size=gs)
            print("[Asym INT] sample:", W_int_asym.view(-1)[:32].tolist())
            print()
            W_fp = quant_datatype(W, wq_bits=bits, datatype=dtype_dir, group_size=gs)
            print(f"[FP{bits} {dtype_dir}] sample:", W_fp.view(-1)[:32].tolist())

        # ─────────────────────────────────────────────────────────────
        #  (D) original per-channel / per-group dequantized value after quant
        # ─────────────────────────────────────────────────────────────
        if is_compute and len(dims) == 2 and grouping:
            W = torch.from_numpy(arr).float()
            gw = gs if grouping == 'per-group' else None

            # choose int vs fp quant API
            if dtype_dir.startswith("int"):
                if "asym" in dtype_dir:
                    Q = quant_int_asym(W, wq_bits=bits, group_size=gw)
                else:
                    Q = quant_int(W,      wq_bits=bits, group_size=gw)
            else:
                Q = quant_datatype(W, wq_bits=bits,
                                   datatype=dtype_dir,
                                   group_size=gw)

            if grouping == 'per-channel':
                for ch in range(Q.shape[0]):
                    sample = Q[ch, :32].tolist()
                    print(f"Ch{ch:4d} Qvals: {sample}…")
                    if ch == 4:
                        break
                print()
            else:
                # per-group: reshape [K, C] → [K, G, gs] 
                G = Q.shape[1] // gs
                R = Q.reshape(Q.shape[0], G, gs)
                K = R.shape[0]
                for ch in range(min(K, 4)):
                    for g in range(min(G, 5)):
                        sample = R[ch, g, :32].tolist()
                        print(f"Ch{ch:4d} Gr{g:4d} Qvals: {sample}…")
                print()
            print()

        stop_name = ("model.decoder.layers.0.self_attn.k_proj.weight_scale"   
                     if "fp16_baseline" not in model_path
                     else "model.decoder.layers.0.self_attn.k_proj.weight")
        #---- TinyLlama, phi-2b ----
        #"model.layers.0.self_attn.q_proj.weight_zp"  
        #"model.layers.0.self_attn.q_proj.weight_scale" 
        
        #---- OPT-1.3B ----
        #"model.decoder.layers.0.self_attn.k_proj.weight_scale" 
        #"model.decoder.layers.0.self_attn.k_proj.weight_zp"
        if name == stop_name:
            break

if __name__ == "__main__":
    model_path = "onnx_graph/facebook__opt-1.3b/per-channel/fp4/w_4_gs_none/model.onnx"
    
    '''            --------baseline---------
    "onnx_graph/facebook__opt-1.3b/fp32_baseline/model.onnx"
    "onnx_graph/facebook__opt-1.3b/fp16_baseline/model.onnx"
                   -----bit_width = 4 ------
    "onnx_graph/facebook__opt-1.3b/per-channel/int4/w_4_gs_none/model.onnx"
    "onnx_graph/facebook__opt-1.3b/per-channel/int4_asym/w_4_gs_none/model.onnx"
    "onnx_graph/facebook__opt-1.3b/per-channel/fp4/w_4_gs_none/model.onnx"
    
    "onnx_graph/facebook__opt-1.3b/per-group/int4/w_4_gs_128/model.onnx"
    "onnx_graph/facebook__opt-1.3b/per-group/int4_asym/w_4_gs_128/model.onnx" 
    "onnx_graph/facebook__opt-1.3b/per-group/fp4/w_4_gs_128/model.onnx"
                   -----bit_width = 8 ------
    "onnx_graph/facebook__opt-1.3b/per-channel/int8/w_8_gs_none/model.onnx"
    "onnx_graph/facebook__opt-1.3b/per-channel/int8_asym/w_8_gs_none/model.onnx"
    "onnx_graph/facebook__opt-1.3b/per-channel/fp8_e2m5/w_8_gs_none/model.onnx"
    "onnx_graph/facebook__opt-1.3b/per-channel/fp8_e3m4/w_8_gs_none/model.onnx"
    "onnx_graph/facebook__opt-1.3b/per-channel/fp8_e4m3/w_8_gs_none/model.onnx"
    "onnx_graph/facebook__opt-1.3b/per-channel/fp8_e5m2/w_8_gs_none/model.onnx"
    
    "onnx_graph/facebook__opt-1.3b/per-group/int8/w_8_gs_128/model.onnx"
    "onnx_graph/facebook__opt-1.3b/per-group/int8_asym/w_8_gs_128/model.onnx" 
    "onnx_graph/facebook__opt-1.3b/per-group/fp8_e2m5/w_8_gs_128/model.onnx"
    "onnx_graph/facebook__opt-1.3b/per-group/fp8_e3m4/w_8_gs_128/model.onnx"
    "onnx_graph/facebook__opt-1.3b/per-group/fp8_e4m3/w_8_gs_128/model.onnx"
    "onnx_graph/facebook__opt-1.3b/per-group/fp8_e5m2/w_8_gs_128/model.onnx"
    '''
    
    
    
    
    
    # "onnx_graph/TinyLlama__TinyLlama-1.1B-Chat-v1.0/per-channel/fp8_e5m2/w_8_gs_none/model.onnx"
    # "onnx_graph/TinyLlama__TinyLlama-1.1B-Chat-v1.0/per-group/fp8_e5m2/w_8_gs_128/model.onnx"
    analyze_quant_initializers(model_path)
