# Language Model Quantization and Analysis Framework

This project provides a comprehensive framework for quantizing HuggingFace language models, evaluating their performance, and conducting detailed error analysis. It allows users to experiment with various quantization techniques, including different bit widths, data types (INT, FP, MX), and grouping strategies (per-channel, per-group).

The core workflow consists of two main stages:
1.  **Export & Quantize**: A PyTorch model is loaded from HuggingFace, its weights are quantized, and the model is exported to the ONNX format.
2.  **Evaluate**: The quantized ONNX model is evaluated on a standard dataset (e.g., wikitext) to measure its perplexity (PPL).

The framework also includes advanced tools for fine-grained analysis of quantization error, such as Mean Squared Error (MSE) and element-wise error comparisons.

## Directory Structure

-   `onnx_graph/`: This is the default output directory where all generated ONNX models are stored. The structure is organized by model name, quantization granularity, data type, and parameters.
-   `quant_utils/`: A library containing the core quantization logic. `quant_weight.py` implements various quantization algorithms (symmetric/asymmetric INT, custom FP formats, etc.).
-   `MSE&P99/`: Contains scripts for analyzing the Mean Squared Error (MSE) and P99 (99th percentile) error of quantized weights against the original FP16 weights.
-   `elwise_error/`: Contains scripts for performing element-wise error analysis between original and quantized models.
-   `ort_quant/`: Contains scripts related to ONNX Runtime quantization and evaluation, specifically for models like TinyLlama.
-   `results_quant/`: The default directory for HuggingFace model downloads and cache, as configured in the shell scripts.

## Core Scripts and Usage

This section describes the main workflow for quantizing and evaluating a model.

### Core Scripts

-   `export_and_quant.py`: The main script for loading a HuggingFace model, applying weight quantization, and exporting it to the ONNX format.
    -   `--model_name`: The name of the model on HuggingFace Hub (e.g., `TinyLlama/TinyLlama-1.1B-Chat-v1.0`).
    -   `--wq_bits`: The bit width for quantization (e.g., 4, 8).
    -   `--wq_datatype`: The target data type (e.g., `int4`, `fp4`, `int8_asym`).
    -   `--wq_groupsize`: The group size for quantization. A value of `128` means per-group quantization, while `-1` or `0` means per-channel.
-   `eval.py`: The script for evaluating the perplexity (PPL) of a quantized ONNX model on the wikitext dataset. It uses the same arguments as `export_and_quant.py` to locate the correct ONNX model file.

### Example Workflow

Here is a typical workflow for quantizing and evaluating a model.

**Step 1: Export and Quantize the Model**

First, run `export_and_quant.py` to create a quantized ONNX model. The following command quantizes the TinyLlama 1.1B model to 4-bit asymmetric integer (`int4_asym`) with a group size of 128.

```bash
python export_and_quant.py \\
    --model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \\
    --wq_bits 4 \\
    --wq_datatype "int4_asym" \\
    --wq_groupsize 128
```

The script will download the model, apply quantization, and save the output to:
`onnx_graph/TinyLlama__TinyLlama-1.1B-Chat-v1.0/per-group/int4_asym/w_4_gs_128/model.onnx`

**Step 2: Evaluate the Quantized Model**

Next, run `eval.py` with the same parameters to evaluate the perplexity of the model you just created.

```bash
python eval.py \\
    --model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \\
    --wq_bits 4 \\
    --wq_datatype "int4_asym" \\
    --wq_groupsize 128
```

The script will load the specified ONNX model, run it on the wikitext test set, and print the final perplexity score.

## Automation

The project includes several shell scripts (`.sh`) to automate the quantization and evaluation process. For example, `export_and_quant.sh` can be configured to loop through multiple models, bit widths, and data types, running `export_and_quant.py` for each combination. This is useful for large-scale experiments.
