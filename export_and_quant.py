# export_and_quant.py

import os, argparse
import torch
import onnx
from onnx import save_model
from onnx.external_data_helper import convert_model_to_external_data
from transformers import AutoModelForCausalLM, AutoTokenizer
from quant_utils.quant_weight import quant_model


def parse_args():
    p = argparse.ArgumentParser(
        description="Export & quantize a causal-LM into ONNX with BitMoD quant."
    )
    p.add_argument("--model_name", type=str, required=True,
                   help="HuggingFace model name, e.g., TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--wq_bits", type=int, default=None,
                   help="bit width (3-8). If omitted and no --wq_datatype, baseline export only.")
    p.add_argument("--wq_datatype", type=str, default=None,
                   help="datatype, e.g., int4, int4_asym, fp4, fp8_e5m2, bf16")
    p.add_argument("--baseline_datatype", type=str, default="fp16",
                   choices=["fp16", "bf16", "fp32"],
                   help="baseline export dtype when --wq_datatype is omitted (default: fp16)")
    p.add_argument("--wq_groupsize", type=int, default=None,
                   help="granularity: -1=per-tensor, 0/None=per-channel, >0=per-group (group size)")
    return p.parse_args()


def main():
    args = parse_args()
    MODEL = args.model_name

    dtype = args.wq_datatype                   # None → baseline export
    groupsize = args.wq_groupsize if dtype else None
    baseline = args.baseline_datatype          # "fp16" or "bf16 or "fp32"

    # 1) output directory structure
    #    onnx_graph/<model_slug>/{per-tensor,per-channel,per-group}/<dtype>/w_<bits>_gs_<size>
    #    or onnx_graph/<model_slug>/<fp16_baseline|fp32_baseline>
    root = "onnx_graph"
    model_slug = MODEL.replace("/", "__")
    model_dir = os.path.join(root, model_slug)

    if dtype is None:
        # baseline
        out_dir = os.path.join(model_dir, f"{baseline}_baseline")
    else:
        gs = 0 if groupsize is None else groupsize
        if gs == -1:
            grouping = "per-tensor"
            gs_name = "tensor"
        elif gs > 0:
            grouping = "per-group"
            gs_name = str(gs)
        else:
            grouping = "per-channel"
            gs_name = "none"
        out_dir = os.path.join(
            model_dir,
            grouping,
            dtype,
            f"w_{args.wq_bits}_gs_{gs_name}"
        )

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "model.onnx")

    # 2) model load / cast
    if dtype is None:
        # baseline dtype
        if baseline == "fp16":
            model = AutoModelForCausalLM.from_pretrained(
                MODEL,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="cpu",
                trust_remote_code=True,
                attn_implementation="eager",
            )
        elif baseline == "bf16":
            model = AutoModelForCausalLM.from_pretrained(
                MODEL,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                device_map="cpu",
                trust_remote_code=True,
                attn_implementation="eager",
            )
        else:  # fp32
            model = AutoModelForCausalLM.from_pretrained(
                MODEL,
                low_cpu_mem_usage=True,
                device_map="cpu",
                trust_remote_code=True,
                attn_implementation="eager",
            )
    else:
        # Quantization path
        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            low_cpu_mem_usage=True,
            device_map="cpu",
            trust_remote_code=True,
            attn_implementation="eager",
        )
        # NOTE: keep fp32 for fp8_e5m2, else use fp16 for efficiency
        model = model.float() if dtype == "fp8_e5m2" else model.half()

    # 3) apply quantization (if requested)
    if dtype is not None:
        quant_model(model, args.wq_bits, dtype, groupsize)

    model.eval()

    # 4) dummy input (dynamic axes)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, use_fast=False, trust_remote_code=True
    )
    dummy = tokenizer("Hello, world!", return_tensors="pt").input_ids

    model.config.use_cache = False  

    import torch.nn as nn
    class OnlyLogits(nn.Module):
        def __init__(self, m): 
            super().__init__()
            self.m = m
        def forward(self, input_ids):
            out = self.m(input_ids=input_ids, use_cache=False, return_dict=True)
            return out.logits

    wrapper = OnlyLogits(model)
    
    # 5) ONNX export
    torch.onnx.export(
        wrapper,
        (dummy,),
        out_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
        "input_ids": {0: "batch", 1: "seq_len"},  
        "logits":    {0: "batch", 1: "seq_len"},
        },
        opset_version=13,
        do_constant_folding=False,
    )
    model_onnx = onnx.load(out_path)
    convert_model_to_external_data(
        model_onnx,
        all_tensors_to_one_file=True,
        location="model.onnx.data",
        size_threshold=1024,
    )
    save_model(model_onnx, out_path)

    label = dtype if dtype is not None else baseline
    print(f"[Export] {label} → {out_path}")


if __name__ == "__main__":
    main()

'''
# Models 
facebook/opt-1.3b microsoft/phi-2 TinyLlama/TinyLlama-1.1B-Chat-v1.0


# Baselines
export HF_HOME="results_quant"
python export_and_quant.py --model_name ${model} # default fp16 baseline
python export_and_quant.py --model_name ${model} --baseline_datatype fp32 # fp32 baseline
python export_and_quant.py --model_name ${model} --baseline_datatype bf16 # bf16 baseline


# Quantized datatype (weight-only)
python export_and_quant.py \
--model_name ${model} \
--wq_bits ${wq_bit} \
--wq_datatype "${dtype}" \
--wq_groupsize ${gs}
# where ${dtype} ∈ {int8, int8_asym, fp8_e2m5, fp8_e3m4, fp8_e4m3, fp8_e5m2, fp4, int4, int4_asym}
# Note: bf16 is a baseline dtype, not a weight-quant datatype
'''