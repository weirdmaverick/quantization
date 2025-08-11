# export_and_quant.py

import os, argparse
import torch
import onnx 
from onnx import save_model
from onnx.external_data_helper import convert_model_to_external_data
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from quant_utils.quant_weight import quant_model
from optimum.onnxruntime import ORTModelForCausalLM

def parse_args():
    p = argparse.ArgumentParser(
        description="Export & quantize a causal-LM into ONNX with BitMoD quant."
    )
    p.add_argument("--model_name", type=str, required=True,
                   help="HuggingFace model name, example: meta-llama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--wq_bits", type=int, default=None,
                   help="bit width (3-8). else, only create fp16 baseline")
    p.add_argument("--wq_datatype", type=str, default=None,
                   help="datatype, example: int4, int4_asym, fp4")
    p.add_argument("--baseline_datatype", type=str, default=None,
                   help="baseline datatype, example: fp16, fp32")
    p.add_argument("--wq_groupsize", type=int, default=128)
    return p.parse_args()

def main():
    args = parse_args()
    MODEL = args.model_name
    # 1) case selection
    cfg = {
        "name": args.baseline_datatype,
        "bits": args.wq_bits or 16 or 32,
        "dtype": args.wq_datatype,
        "groupsize": args.wq_groupsize if args.wq_datatype else None
    }

    # 2) output directory structure
    #    onnx_graph/<model_slug>/{per-group,per-channel}/<dtype>/w_<bits>_gs_<size> or baseline
    root = "onnx_graph"
    model_slug = MODEL.replace("/", "__")
    model_dir = os.path.join(root, model_slug)

    if cfg["dtype"] is None:
        # fp16 or fp32 baseline
        if cfg["bits"] == 16 :
            out_dir = os.path.join(model_dir, "fp16_baseline")
        elif cfg["bits"] == 32 :
            out_dir = os.path.join(model_dir, "fp32_baseline")
    else:
        # granularity 
        gs = cfg["groupsize"] if cfg["groupsize"] is not None else -1
        grouping = "per-group" if gs > 0 else "per-channel"
        # group size
        gs_name = str(gs) if gs > 0 else "none"
        # datatype (ex: int4_asym, int4_sym, fp4 등)
        dtype = cfg["dtype"]
        bits  = cfg["bits"]
        # directory: grouping / dtype / w_<bits>_gs_<gs_name>
        out_dir = os.path.join(
            model_dir,
            grouping,
            dtype,
            f"w_{bits}_gs_{gs_name}"
        )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "model.onnx")

    # 3) model load
    if cfg["name"] == "fp16":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="cpu",
            trust_remote_code =True,
            attn_implementation = "eager"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            low_cpu_mem_usage=True,
            device_map="cpu",
            trust_remote_code =True,
            attn_implementation = "eager"
        )
    if cfg["dtype"] == "fp8_e5m2":   # overflow issue is occured if fp8_e5m2 is casted to fp16
        model = model.float()
    else:
        model = model.half()
    
    # 4) quantization
    if cfg["dtype"] is not None:
        quant_model(model, cfg["bits"], cfg["dtype"], cfg["groupsize"])
    
    # model = model.float() 
    model.eval()

    # 5) dummy input (dynamic axes)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, use_fast=False, trust_remote_code=True   # use_fast ? pure python tokenizer : rust based tokenizer
    )
    dummy = tokenizer("Hello, world!", return_tensors="pt").input_ids

    # 6) ONNX export
    out_path = os.path.join(out_dir, "model.onnx")
    torch.onnx.export(
        model,
        (dummy,),
        out_path,
        input_names=["input_ids"], 
        output_names=["logits"],
        dynamic_axes={"input_ids": {0:"batch",1:"seq_len"},
                      "logits":    {0:"batch",1:"seq_len"}},
        opset_version=17, # 20 for gold server
        do_constant_folding=False
    )
    model_onnx = onnx.load(out_path)
    convert_model_to_external_data(
        model_onnx,
        all_tensors_to_one_file=True,
        location="model.onnx.data",
        size_threshold=1024
    )

    save_model(model_onnx, out_path)
    if "bits" == 16:
        selected_dtype ="name"
    else:
        selected_dtype ="dtype" 
    print(f"[Export] {selected_dtype} → {out_path}")
    

if __name__ == "__main__":
    main()
