import os
import argparse
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM
from quant_utils.quant_weight import quant_model

def parse_args():
    p = argparse.ArgumentParser(
        description="Export & quantize a causal-LM into ONNX with BitMoD quant."
    )
    p.add_argument("--model_name", type=str, required=True,
                   help="HuggingFace model name, e.g. meta-llama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--wq_bits", type=int, default=None,
                   help="bit width (3-8). else, only create fp32 baseline")
    p.add_argument("--wq_datatype", type=str, default=None,
                   help="datatype, example: int4, int4_asym, fp4. else, fp32 baseline")
    p.add_argument("--wq_groupsize", type=int, default=128)
    return p.parse_args()

def main():
    args = parse_args()
    MODEL = args.model_name

    # 1) case selection
    cfg = {
        "name": (args.wq_datatype or "fp16"),
        "bits": args.wq_bits or 32,
        "dtype": args.wq_datatype,
        "groupsize": args.wq_groupsize if args.wq_datatype else None
    }

    # 2) output directory structure
    root       = "onnx_graph"
    model_slug = MODEL.replace("/", "__")
    model_dir  = os.path.join(root, model_slug)

    if cfg["dtype"] is None:
        out_dir = os.path.join(model_dir, "baseline")
    else:
        gs       = cfg["groupsize"] if cfg["groupsize"] is not None else -1
        grouping = "per-group" if gs > 0 else "per-channel"
        gs_name  = str(gs) if gs > 0 else "none"
        dtype    = cfg["dtype"]
        bits     = cfg["bits"]
        out_dir  = os.path.join(model_dir, grouping, dtype, f"w_{bits}_gs_{gs_name}")

    os.makedirs(out_dir, exist_ok=True)

    # 3) model load
    if cfg["name"] == "fp16":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="cpu",
            trust_remote_code=True,
            attn_implementation="eager"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            low_cpu_mem_usage=True,
            device_map="cpu",
            trust_remote_code=True,
            attn_implementation="eager"
        )

    # 4) quantization (BitMoD)
    if cfg["dtype"] is not None:
        quant_model(model, cfg["bits"], cfg["dtype"], cfg["groupsize"])

    model = model.float()
    model.eval()

    # 5) tokenizer + dummy input
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, use_fast=False, trust_remote_code=True
    )
    dummy = tokenizer("Hello, world!", return_tensors="pt").input_ids

    # 6) torch.onnx.export → model.onnx
    onnx_path = os.path.join(out_dir, "model.onnx")
    torch.onnx.export(
        model,
        (dummy,),
        onnx_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0:"batch",1:"seq_len"},
                      "logits":    {0:"batch",1:"seq_len"}},
        opset_version=20,
        do_constant_folding=False
    )
    AutoConfig.from_pretrained(MODEL).save_pretrained(out_dir)

    # 7) Optimum ORTModelForCausalLM 로드 후 external data 포맷으로 재저장
    ort_model = ORTModelForCausalLM.from_pretrained(
        out_dir,                
        file_name="model.onnx", 
        from_transformers=False, 
        library="transformers",
        use_cache=False,
        use_io_binding=False
    )
    
    ort_save_dir = os.path.join(out_dir, "ort")
    os.makedirs(ort_save_dir, exist_ok=True)
    
    
    ort_model.save_pretrained(
        ort_save_dir,
        use_external_data_format=True 
    )

    # 8) tokenizer & config 파일도 함께 저장
    tokenizer.save_pretrained(ort_save_dir)

    print(f"[Export] {cfg['name']} → {out_dir} and external data → {ort_save_dir}")

if __name__ == "__main__":
    main()
