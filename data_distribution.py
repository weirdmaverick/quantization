import os
import numpy as np
import matplotlib.pyplot as plt

# —————————————————————————————————————————————————————————
# directory & file 
# —————————————————————————————————————————————————————————
TARGET_WEIGHT_FILE = "model.decoder.layers.0.self_attn.k_proj.weight"  

MODEL_DIRS = {
    "fp16_baseline": "onnx_graph/facebook__opt-1.3b/fp16_baseline"
}

# 각 모델별로 raw 바이너리 dtype 지정
DTYPES = {name: np.float16 for name in MODEL_DIRS}
DTYPES["fp8_e5m2_per-group"] = np.float32  # 예외 처리

COLORS = {
    "fp16_baseline": "tab:blue",
    "int4_per-channel": "tab:orange",
    "int4_asym_per-channel": "tab:green",
    "fp4_per-channel": "tab:red",
}

LINEWIDTH = 2.5
BINS = 200
FP8_E3M4_RESOLUTION = [
    (0.0080448274, -0.24938965, -0.1287172387),
    (0.0040224137, -0.1287172387, -0.0643586193),
    (0.0020112069, -0.0643586193, -0.0321793097),
    (0.0010056034, -0.0321793097, -0.0160896548),
    (0.0005028017, -0.0160896548, -0.0080448274),
    (0.0002514009, -0.0080448274, -0.0040224137),
    (0.0001257004, -0.0040224137, -0.0020112069),
    (6.28502e-05,  -0.0020112069, -0.0010684536),
    (0.0010684536, -0.0010684536,  0.0010684536),  # zero 주변
    (6.28502e-05,   0.0010684536,  0.0020112069),
    (0.0001257004,  0.0020112069,  0.0040224137),
    (0.0002514009,  0.0040224137,  0.0080448274),
    (0.0005028017,  0.0080448274,  0.0160896548),
    (0.0010056034,  0.0160896548,  0.0321793097),
    (0.0020112069,  0.0321793097,  0.0643586193),
    (0.0040224137,  0.0643586193,  0.1287172387),
    (0.0080448274,  0.1287172387,  0.24938965)
]


# —————————————————————————————————————————————————————————
# 1) 모든 weight 읽어서 flatten
# —————————————————————————————————————————————————————————
weights = {}
for name, d in MODEL_DIRS.items():
    path = os.path.join(d, TARGET_WEIGHT_FILE)
    dtype = DTYPES[name]
    arr = np.fromfile(path, dtype=dtype)
    weights[name] = arr.astype(np.float32)

# —————————————————————————————————————————————————————————
# 2) 분포 그리기 + log scale 분석
# —————————————————————————————————————————————————————————
plt.figure(figsize=(8,5))

for name, arr in weights.items():
    hist, bins = np.histogram(arr, bins=BINS, density=False)
    centers = 0.5*(bins[:-1] + bins[1:])
    total_count = len(arr)

    #  weight의 min/max 출력
    min_val, max_val = arr.min(), arr.max()
    print(f"\n [{name}] Weight Min/Max")
    print(f"   ▶ Min: {min_val:.8f}")
    print(f"   ▶ Max: {max_val:.8f}")
    print(f"   ▶ Total count: {total_count:,}")

    #  log 단위 구간 설정
    abs_arr = np.abs(arr)  # 절댓값 기준
    log_bins = [0, 1e-5, 1e-4, 1e-3, 1e-2, max_val]  # 마지막은 max까지 커버
    print(f"\n [{name}] Log10 based absolute value ratio")

    for i in range(len(log_bins)-1):
        low, high = log_bins[i], log_bins[i+1]
        mask = (abs_arr >= low) & (abs_arr < high)
        count = np.sum(mask)
        ratio = (count / total_count) * 100
        print(f"   {low:.0e} ~ {high:.0e} ➜ {count:,} counts ({ratio:.2f}%)")
    
    Weight_MAX = 0.24938965
    # --------- INT8 precision -------------
    int8_threshold = Weight_MAX/127
    below_mask = abs_arr < int8_threshold
    above_mask = abs_arr > int8_threshold
    below_count = np.sum(below_mask)
    above_count = np.sum(above_mask)
    
    below_ratio = (below_count/total_count) * 100
    above_ratio = (above_count/total_count) * 100
    
    print()
    print(f"\n [INT8] Resolution: {int8_threshold}(fixed)")
    print(f" W   <{int8_threshold} -> {below_count:,} counts ({below_ratio:.2f}%)")
    print(f" W   ≥{int8_threshold} -> {above_count:,} counts ({above_ratio:.2f}%)")
    
    # ratio of error spike range
    print()
    low_limit = 0.0
    mid_limit = 0.0005
    high_limit = 0.002

    range_mask = (abs_arr >= low_limit) & (abs_arr < high_limit)
    range_count = np.sum(range_mask)

    low_mask = (abs_arr >= low_limit) & (abs_arr < mid_limit)
    low_count = np.sum(low_mask)

    mid_mask = (abs_arr >= mid_limit) & (abs_arr < high_limit)
    mid_count = np.sum(mid_mask)

    if range_count > 0:
        low_ratio = (low_count / range_count) * 100
        mid_ratio = (mid_count / range_count) * 100
    else:
        low_ratio, mid_ratio = 0, 0

    print("\n [INT8] [0~0.002(~int8 resolution)] range ratio")
    print(f"   ▶ total count: {range_count:,}")
    print(f"   0 ~ {mid_limit}      ➜ {low_count:,} counts ({low_ratio:.2f}%)")
    print(f"   {mid_limit} ~ {high_limit} ➜ {mid_count:,} counts ({mid_ratio:.2f}%)")
    
    # --------- FP8 E3M4 precision -------------
    print(f"\n [FP8_E3M4] Resolution range ")

    for step, start, end in FP8_E3M4_RESOLUTION:
        if end <= 0:  
            low, high = abs(start), abs(end)
            if low > high:
                low, high = high, low
            mask = (abs_arr >= low) & (abs_arr < high)
            count = np.sum(mask)
            ratio = (count / total_count) * 100
            print(f"   Δ={step:.8f} | {low:.6f} < W < {high:.6f} ➜ {count:,} ({ratio:.2f}%)")

    # resolution between zero and first adjacent element  
    zero_low, zero_high = 0.0, 0.0010684536
    zero_mask = (abs_arr >= zero_low) & (abs_arr < zero_high)
    zero_count = np.sum(zero_mask)
    zero_ratio = (zero_count / total_count) * 100
    print(f"   Δ=0.00106845 | {zero_low:.6f} < W < {zero_high:.6f} ➜ {zero_count:,} ({zero_ratio:.2f}%)")

    for step, start, end in FP8_E3M4_RESOLUTION:
        if start >= 0:  
            low, high = abs(start), abs(end)
            if low > high:
                low, high = high, low
            mask = (abs_arr >= low) & (abs_arr < high)
            count = np.sum(mask)
            ratio = (count / total_count) * 100
            print(f"   Δ={step:.8f} | {low:.6f} < W < {high:.6f} ➜ {count:,} ({ratio:.2f}%)")
    

   

""" plt.xlabel("De-quantized Weight Value", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.title("Per-datatype Weight Distribution (4-bit)", fontsize=14)
plt.legend(loc="upper right", fontsize=9)
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()  """
