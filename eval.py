#!/usr/bin/env python3
import os
import glob
import time
import argparse
import numpy as np
import math
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
from onnxruntime import InferenceSession

def batch_nll_and_count(logits: np.ndarray, ids: np.ndarray, mask: np.ndarray):
    # logits: (1, L, V), ids & mask: (1, L)
    L = logits.shape[1]
    shift_logits = logits[:, :-1, :]     # (1, L-1, V)
    shift_labels = ids[:, 1:]            # (1, L-1)
    shift_mask   = mask[:, 1:]           # (1, L-1)

    # stable log‐softmax
    max_logit = np.max(shift_logits, axis=-1, keepdims=True)
    exp_logits = np.exp(shift_logits - max_logit)
    logsumexp  = np.log(np.sum(exp_logits, axis=-1, keepdims=True)) + max_logit
    log_probs  = shift_logits - logsumexp  # (1, L-1, V)

    # true token log‐probs
    token_log_probs = log_probs[0, np.arange(L-1), shift_labels[0]]
    token_log_probs *= shift_mask[0]

    nll   = -np.sum(token_log_probs)
    ntoks = np.sum(shift_mask)
    return nll, ntoks

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate ONNX model(s) perplexity; choose baseline or quantized by CLI."
    )
    p.add_argument("--model_name",        type=str, required=True,
                   help="HuggingFace model name, e.g. meta-llama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--dataset",           type=str, default="wikitext-2-raw-v1",
                   help="HuggingFace Datasets name (wikitext builder), e.g. wikitext-2-raw-v1")
    p.add_argument("--baseline_datatype", type=str, default="fp16",
                   choices=["fp16","fp32"],
                   help="Baseline datatype if no quant: fp16 or fp32")
    p.add_argument("--wq_bits",           type=int, default=None,
                   help="Weight-quant bit width (3-8); omit → baseline-only")
    p.add_argument("--wq_datatype",       type=str, default=None,
                   help="Quant datatype, e.g. int4, int4_asym, fp4, fp8_e5m2")
    p.add_argument("--wq_groupsize",      type=int, default=128,
                   help="Quant group size; 0 or -1 → per-channel")
    return p.parse_args()

def main():
    args = parse_args()
    MODEL = args.model_name
    DATASET = args.dataset

    # tokenizer + dataset 
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, use_fast=False, trust_remote_code=True
    )
    ds = load_dataset("wikitext", DATASET, split="test")
    enc = tokenizer("\n\n".join(ds["text"]), return_tensors="pt")
    input_ids = enc.input_ids.numpy()    # shape (1, total_tokens)
    seqlen = 1024
    nsamples = input_ids.size // seqlen

    # pick ONNX model(s) based on CLI flags 
    model_slug = MODEL.replace("/", "__")
    root = os.path.join("onnx_graph", model_slug)

    if args.wq_datatype:
        grouping = "per-channel" if args.wq_groupsize in (0,-1) else "per-group"
        gs_name  = "none" if args.wq_groupsize in (0,-1) else str(args.wq_groupsize)
        subdir = os.path.join(root, grouping, args.wq_datatype, f"w_{args.wq_bits}_gs_{gs_name}")
        onnx_paths = [os.path.join(subdir, "model.onnx")]
    else:
        base_dir = "fp16_baseline" if args.baseline_datatype=="fp16" else "fp32_baseline"
        onnx_paths = [os.path.join(root, base_dir, "model.onnx")]

    # ── run evaluation ──
    for onnx_path in onnx_paths:
        cfg_dir = os.path.relpath(os.path.dirname(onnx_path), root)
        print(f"\n[Eval] Configuration: {cfg_dir}")
        sess = InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

        total_nll = 0.0
        total_tok = 0.0

        for i in tqdm(range(nsamples), desc=f"[eval] {cfg_dir}"):
            batch = input_ids[:, i*seqlen:(i+1)*seqlen]
            mask  = np.ones_like(batch, dtype=np.int64)
            logits = sess.run(None, {"input_ids": batch})[0]
            nll, ntoks = batch_nll_and_count(logits, batch, mask)
            total_nll += nll
            total_tok += ntoks

        ppl = math.exp(total_nll / total_tok)
        print(f"{cfg_dir:40s} PPL: {ppl:.3f}")

if __name__ == "__main__":
    main()
