import onnx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ───────────────────────────────────────────────────────────────
# 1) ONNX model file(.onnx) path configuration
# ───────────────────────────────────────────────────────────────
paths = {
    'fp16'     : '../onnx_graph/facebook__opt-1.3b/fp16_baseline/model.onnx',
    # --------------------------------------per-channel--------------------------------------------- 
    # -- bit-width = 8 --
    'int8_asym': '../onnx_graph/facebook__opt-1.3b/per-channel/int8_asym/w_8_gs_none/model.onnx', 
    'int8'     : '../onnx_graph/facebook__opt-1.3b/per-channel/int8/w_8_gs_none/model.onnx',
    'fp8_e3m4' : '../onnx_graph/facebook__opt-1.3b/per-channel/fp8_e3m4/w_8_gs_none/model.onnx', 
    'fp8_e4m3' : '../onnx_graph/facebook__opt-1.3b/per-channel/fp8_e4m3/w_8_gs_none/model.onnx', 
    # -- bit-width = 7 --
    'int7_asym': '../onnx_graph/facebook__opt-1.3b/per-channel/int7_asym/w_7_gs_none/model.onnx',
    'int7'     : '../onnx_graph/facebook__opt-1.3b/per-channel/int7/w_7_gs_none/model.onnx',
    # -- bit-width = 6 --
    'int6_asym': '../onnx_graph/facebook__opt-1.3b/per-channel/int6_asym/w_6_gs_none/model.onnx',
    'int6'     : '../onnx_graph/facebook__opt-1.3b/per-channel/int6/w_6_gs_none/model.onnx',
    'fp6e3m2'  : '../onnx_graph/facebook__opt-1.3b/per-channel/fp6e3m2/w_6_gs_none/model.onnx',
    'fp6e2m3'  : '../onnx_graph/facebook__opt-1.3b/per-channel/fp6e2m3/w_6_gs_none/model.onnx',
    # -- bit-width = 5 --
    'int5_asym': '../onnx_graph/facebook__opt-1.3b/per-channel/int5_asym/w_5_gs_none/model.onnx',
    'int5'     : '../onnx_graph/facebook__opt-1.3b/per-channel/int5/w_5_gs_none/model.onnx',
    'fp5e2m2'  : '../onnx_graph/facebook__opt-1.3b/per-channel/fp5e2m2/w_5_gs_none/model.onnx',
    'fp5e3m1'  : '../onnx_graph/facebook__opt-1.3b/per-channel/fp5e3m1/w_5_gs_none/model.onnx',
    # -- bit-width = 4 --
    'int4_asym': '../onnx_graph/facebook__opt-1.3b/per-channel/int4_asym/w_4_gs_none/model.onnx',
    'int4'     : '../onnx_graph/facebook__opt-1.3b/per-channel/int4/w_4_gs_none/model.onnx',
    'fp4'      : '../onnx_graph/facebook__opt-1.3b/per-channel/fp4/w_4_gs_none/model.onnx',
    # -- bit-width = 3 --
    'int3_asym': '../onnx_graph/facebook__opt-1.3b/per-channel/int3_asym/w_3_gs_none/model.onnx',
    'int3'     : '../onnx_graph/facebook__opt-1.3b/per-channel/int3/w_3_gs_none/model.onnx',
    'fp3'      : '../onnx_graph/facebook__opt-1.3b/per-channel/fp3/w_3_gs_none/model.onnx',
}

# ───────────────────────────────────────────────────────────────
# 2) tensor to extract
# ───────────────────────────────────────────────────────────────
tensor_name = 'model.decoder.layers.0.self_attn.k_proj.weight'

# ───────────────────────────────────────────────────────────────
# 3) initializer to numpy array 
# ───────────────────────────────────────────────────────────────
def load_weight(onnx_path: str, tensor_name: str) -> np.ndarray:
    model = onnx.load(onnx_path)
    for init in model.graph.initializer:
        if init.name == tensor_name:
            return onnx.numpy_helper.to_array(init)
    raise KeyError(f"Initializer '{tensor_name}' not found in {onnx_path}")

# ───────────────────────────────────────────────────────────────
# 4) weight load
# ───────────────────────────────────────────────────────────────
w_fp16      = load_weight(paths['fp16'], tensor_name)

w_int8_asym = load_weight(paths['int8_asym'], tensor_name)
w_int8      = load_weight(paths['int8'], tensor_name)
w_fp8_e3m4  = load_weight(paths['fp8_e3m4'], tensor_name)
w_fp8_e4m3  = load_weight(paths['fp8_e4m3'], tensor_name)

w_int4_asym = load_weight(paths['int4_asym'], tensor_name)
w_int4      = load_weight(paths['int4'], tensor_name)
w_fp4       = load_weight(paths['fp4'], tensor_name)
# ───────────────────────────────────────────────────────────────
# 5) element-wise quantization error 계산 (flatten)
# ───────────────────────────────────────────────────────────────
err_int8_asym = (w_fp16 - w_int8_asym).flatten()
err_int8      = (w_fp16 - w_int8).flatten()
err_fp8_e3m4  = (w_fp16 - w_fp8_e3m4).flatten()
err_fp8_e4m3  = (w_fp16 - w_fp8_e4m3).flatten()

err_int4_asym = (w_fp16 - w_int4_asym).flatten()
err_int4      = (w_fp16 - w_int4).flatten()
err_fp4       = (w_fp16 - w_fp4).flatten()

abs_err_int8_asym = np.abs(err_int8_asym)
abs_err_int8      = np.abs(err_int8)
abs_err_fp8_e3m4  = np.abs(err_fp8_e3m4)
abs_err_fp8_e4m3  = np.abs(err_fp8_e4m3)

abs_err_int4_asym = np.abs(err_int4_asym)
abs_err_int4      = np.abs(err_int4)
abs_err_fp4       = np.abs(err_fp4)


#------------------ P99 ---------------------------
""" p99_int8 = np.percentile(abs_err_int8,99)
print("-----p99_int8-----")
print(p99_int8)
print("-----p99_fpe3m4-----")
p99_fpe3m4 = np.percentile(abs_err_fp8_e3m4,99)
print(p99_fpe3m4)
print("-----p99_fpe4m3-----")
p99_fpe4m3 = np.percentile(abs_err_fp8_e4m3,99)
print(p99_fpe4m3) """

# ───────────────────────────────────────────────────────────────
# 6) index vs. error plot
# ───────────────────────────────────────────────────────────────

# --------------- absolute error plot ----------------------
indices = np.arange(err_int8_asym.shape[0], dtype = int)
#indices = np.arange(err_int8.shape[0], dtype = int)
#indices = np.arange(err_fp8_e3m4.shape[0], dtype = int)
#indices = np.arange(err_fp8_e4m3.shape[0], dtype = int)

#indices = np.arange(err_int4_asym.shape[0], dtype = int)
#indices = np.arange(err_int4.shape[0], dtype = int)
#indices = np.arange(err_fp4.shape[0], dtype = int)

fig, ax = plt.subplots(figsize=(10, 6))
#ax.plot(indices, abs_err_int8_asym, lw = 2.0, label = '|INT8-asym error|')
#ax.plot(indices, abs_err_int8, lw = 2.0, label = '|INT8-sym error|')
#ax.plot(indices, abs_err_fp8_e3m4, lw = 2.0, label = '|FP8_E3M4 error|')
#ax.plot(indices, abs_err_fp8_e4m3, lw = 2.0, label = '|FP8_E4M3 error|')

#ax.plot(indices, abs_err_fp4, lw = 2.0, label = '|FP4 error|')
#ax.plot(indices, abs_err_int4, lw = 2.0, label = '|INT4-sym error|')
ax.plot(indices, abs_err_int4_asym, lw = 2.0, label = '|INT4-asym error|')

ax.set_xlim(0, 100)
ax.set_xticks(np.arange(0, 100, 10))
ax.xaxis.set_major_locator(MultipleLocator(10))
  
# 8-bit plot range
ax.set_ylim(0, 0.0005) 
ax.set_yticks(np.arange(0, 0.0005, 0.000125)) # 8-bit
ax.yaxis.set_major_locator(MultipleLocator(0.000125))       

# 4-bit plot range
""" ax.set_ylim(0, 0.006) 
ax.set_yticks(np.arange(0, 0.006, 0.0005)) # 4-bit  
ax.yaxis.set_major_locator(MultipleLocator(0.0005)) """

ax.grid(which='major', linestyle='-',  linewidth=0.5, alpha=0.7)
ax.grid(which='minor', linestyle='--', linewidth=0.3, alpha=0.5)

#------ error sum ---------
""" sum_int8_sym = sum(abs_err_int8)
print("error_sum_int8")
print(sum_int8_sym)
print("----------------------------")
sum_fp8_e3m4 = sum(abs_err_fp8_e3m4)
print("error_sum_fp8_e3m4")
print(sum_fp8_e3m4)
print("----------------------------")
sum_fp8_e4m3 = sum(abs_err_fp8_e4m3)
print("error_sum_fp8_e4m3")
print(sum_fp8_e4m3)
print("----------------------------")
print("----------------------------")
sum_fp4 = sum(abs_err_fp4)
print("error_sum_fp4")
print(sum_fp4)
print("----------------------------")
sum_int4_sym = sum(abs_err_int4)
print("error_sum_in4_sym")
print(sum_int4_sym)
print("----------------------------")
sum_int4_asym = sum(abs_err_int4_asym)
print("error_sum_int4_asym")
print(sum_int4_asym) """


# -------------------- raw data plot -------------------------------

#plt.plot(indices, err_int8_asym, lw=0.5, label='INT8_asym error')
#plt.plot(indices, err_int8, lw=0.5, label='INT8_sym error')
#plt.plot(indices, err_fp8_e3m4, lw=0.5, label='FP8_E3M4 error')
#plt.plot(indices, err_fp8_e4m3, lw=0.5, label='FP8_E4M3 error')

#plt.plot(indices, err_int4_asym, lw=0.5, label='INT4_asym error')
#plt.plot(indices, err_int4, lw=0.5, label='INT4_sym error')
#plt.plot(indices,  err_fp4, lw=0.5, label='FP4 error')

""" plt.xlabel('Element index')
#plt.ylabel('Quantization Error')
plt.ylabel('Absolute Quantization Error')
plt.title('Quantization Error per Weight Element')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
 """
 
#---------histogram-----------
# 4-bit 
""" data = {
    'FP4'        : abs_err_fp4,
    'INT4-asym'  : abs_err_int4_asym,
    'INT4'       : abs_err_int4,
}
bins = np.linspace(0, 0.01, 200)
h = {}
for lbl, arr in data.items():
    counts, _ = np.histogram(arr, bins=bins)
    h[lbl] = counts
centers = (bins[:-1] + bins[1:]) / 2
plt.figure(figsize=(6,4))
for lbl, counts in h.items():
    plt.step(centers, counts, where='mid', label=lbl, linewidth=1.5) """

# 8-bit 
""" data = {
    'INT8'       : abs_err_int8,
    'FP8_E3M4'   : abs_err_fp8_e3m4,
    'FP8_E4M3'   : abs_err_fp8_e4m3
}
bins = np.linspace(0, 0.0015, 200)
h = {}
for lbl, arr in data.items():
    counts, _ = np.histogram(arr, bins=bins)
    h[lbl] = counts
centers = (bins[:-1] + bins[1:]) / 2
plt.figure(figsize=(6,4))
for lbl, counts in h.items():
    plt.step(centers, counts, where='mid', label=lbl, linewidth=1.5)
"""   
plt.xlabel('Element ID',fontsize=20)
plt.ylabel('Absolute error',fontsize=20)
plt.xticks()
plt.yticks()
plt.legend(fontsize = 20, loc='upper right')
#plt.title('Error Distribution',fontsize=15)
plt.show()   

