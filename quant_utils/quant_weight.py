import types
import torch.nn.functional as F
import torch
import torch.nn as nn
from typing import Optional
import time


#################################  3-bit Datatypes  #################################
INT3 = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
FP3 = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]
FP3_ER_POS = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]  # 3
FP3_ER_NEG = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]  # -3
FP3_EA_POS = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 6.0]  # 6
FP3_EA_NEG = [-6.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]  # -6

#################################  4-bit Datatypes  #################################
INT4 = [-7.0, -6.0, -5.0, -4.0, -3.0, -2.0, 
        -1.0, 0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
FLINT4 = [-16.0, -8.0, -6.0, -4.0, -3.0, -2.0, 
          -1.0, 0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 16.0]
FP4_E2M1 = [-12.0, -8.0, -6.0, -4.0, -3.0, -2.0, 
        -1.0, 0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
FP4_ER_POS = [-12.0, -8.0, -6.0, -4.0, -3.0, -2.0, 
              -1.0, 0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0]
FP4_ER_NEG = [-12.0, -10.0, -8.0, -6.0, -4.0, -3.0, 
              -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
FP4_EA_POS = [-12.0, -8.0, -6.0, -4.0, -3.0, -2.0, 
              -1.0, 0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]
FP4_EA_NEG = [-16.0, -12.0, -8.0, -6.0, -4.0, -3.0, 
              -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]

#################################  5-bit Datatypes  #################################
INT5 = [-15.0, -14.0, -13.0, -12.0, -11.0, -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, 
        -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
FLINT5 = [-64.0, -32.0, -24.0, -16.0, -14.0, -12.0, -10.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, 
          -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0, 24.0, 32.0, 64.0]
FP5_E2M2 = [-28.0, -24.0, -20.0, -16.0, -14.0, -12.0, -10.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, 
            -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 24.0, 28.0]
FP5_E3M1 = [-192.0, -128.0, -96.0, -64.0, -48.0, -32.0, -24.0, -16.0, -12.0, -8.0, -6.0, -4.0, -3.0, 
            -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 128.0, 192.0]

#################################  6-bit Datatypes  #################################
INT6 = [
    -31.0, -30.0, -29.0, -28.0, -27.0, -26.0, -25.0, -24.0, -23.0, -22.0, -21.0, -20.0, -19.0, -18.0, -17.0, 
    -16.0, -15.0, -14.0, -13.0, -12.0, -11.0, -10.0, 
    -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0,
    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0
]
FP6_E2M3 = [
    -60.0, -56.0, -52.0, -48.0, -44.0, -40.0, -36.0, -32.0, -30.0, -28.0, -26.0, -24.0, -22.0, -20.0, -18.0, 
    -16.0, -15.0, -14.0, -13.0, -12.0, -11.0, -10.0, 
    -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0,
    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 36.0, 40.0, 44.0, 48.0, 52.0, 56.0, 60.0
]
FP6_E3M2 = [
    -448.0, -384.0, -320.0, -256.0, -224.0, -192.0, -160.0, -128.0, -112.0, -96.0, -80.0, -64.0, -56.0, -48.0, 
    -40.0, -32.0, -28.0, -24.0, -20.0, -16.0, -14.0, -12.0, 
    -10.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0,
    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 24.0, 28.0, 32.0, 40.0, 48.0, 56.0, 64.0, 80.0, 96.0, 112.0, 128.0, 160.0, 192.0, 224.0, 256.0, 320.0, 384.0, 448.0
]
# -----------------------------------------------------------------------------------------
#################################  8-bit Datatypes  #################################
INT8 = list(range(-127, 128))  # -127,-126,-125, ..., 0, 1, 2, ..., 126, 127
FP8_E2M5 = [
    -7.875, -7.75, -7.625, -7.5, -7.375, -7.25, -7.125, -7.0, 
    -6.875, -6.75, -6.625, -6.5, -6.375, -6.25, -6.125, -6.0, -5.875,
    -5.75, -5.625, -5.5, -5.375, -5.25, -5.125, -5.0, -4.875, -4.75, 
    -4.625, -4.5, -4.375, -4.25, -4.125, -4.0, -3.9375, -3.875, -3.8125,
    -3.75, -3.6875, -3.625, -3.5625, -3.5, -3.4375, -3.375, -3.3125, 
    -3.25, -3.1875, -3.125, -3.0625, -3.0, -2.9375, -2.875, -2.8125,
    -2.75, -2.6875, -2.625, -2.5625, -2.5, -2.4375, -2.375, -2.3125, 
    -2.25, -2.1875, -2.125, -2.0625, -2.0, -1.96875, -1.9375, -1.90625,
    -1.875, -1.84375, -1.8125, -1.78125, -1.75, -1.71875, -1.6875, 
    -1.65625, -1.625, -1.59375, -1.5625, -1.53125, -1.5, -1.46875, -1.4375,
    -1.40625, -1.375, -1.34375, -1.3125, -1.28125, -1.25, -1.21875, 
    -1.1875, -1.15625, -1.125, -1.09375, -1.0625, -1.03125, -1.0, -0.984375,
    -0.96875, -0.953125, -0.9375, -0.921875, -0.90625, -0.890625, 
    -0.875, -0.859375, -0.84375, -0.828125, -0.8125, -0.796875, -0.78125,
    -0.765625, -0.75, -0.734375, -0.71875, -0.703125, -0.6875, -0.671875, 
    -0.65625, -0.640625, -0.625, -0.609375, -0.59375, -0.578125, -0.5625,
    -0.546875, -0.53125, -0.515625, 0.0,
    0.515625, 0.53125, 0.546875, 0.5625, 0.578125, 0.59375, 0.609375, 0.625, 0.640625, 0.65625, 0.671875, 0.6875, 0.703125, 0.71875, 0.734375, 0.75,
    0.765625, 0.78125, 0.796875, 0.8125, 0.828125, 0.84375, 0.859375, 0.875, 0.890625, 0.90625, 0.921875, 0.9375, 0.953125, 0.96875, 0.984375, 1.0,
    1.03125, 1.0625, 1.09375, 1.125, 1.15625, 1.1875, 1.21875, 1.25, 1.28125, 1.3125, 1.34375, 1.375, 1.40625, 1.4375, 1.46875, 1.5, 1.53125, 1.5625,
    1.59375, 1.625, 1.65625, 1.6875, 1.71875, 1.75, 1.78125, 1.8125, 1.84375, 1.875, 1.90625, 1.9375, 1.96875, 2.0, 2.0625, 2.125, 2.1875, 2.25, 2.3125,
    2.375, 2.4375, 2.5, 2.5625, 2.625, 2.6875, 2.75, 2.8125, 2.875, 2.9375, 3.0, 3.0625, 3.125, 3.1875, 3.25, 3.3125, 3.375, 3.4375, 3.5, 3.5625, 3.625,
    3.6875, 3.75, 3.8125, 3.875, 3.9375, 4.0, 4.125, 4.25, 4.375, 4.5, 4.625, 4.75, 4.875, 5.0, 5.125, 5.25, 5.375, 5.5, 5.625, 5.75, 5.875, 6.0, 6.125,
    6.25, 6.375, 6.5, 6.625, 6.75, 6.875, 7.0, 7.125, 7.25, 7.375, 7.5, 7.625, 7.75, 7.875
]
FP8_E3M4 = [
    -31.0, -30.0, -29.0, -28.0, -27.0, -26.0, -25.0, -24.0, -23.0, -22.0, 
    -21.0, -20.0, -19.0, -18.0, -17.0, -16.0, -15.5, -15.0, -14.5, -14.0, -13.5,
    -13.0, -12.5, -12.0, -11.5, -11.0, -10.5, -10.0, -9.5, -9.0, -8.5, -8.0,
    -7.75, -7.5, -7.25, -7.0, -6.75, -6.5, -6.25, -6.0, -5.75, -5.5, -5.25,
    -5.0, -4.75, -4.5, -4.25, -4.0, -3.875, -3.75, -3.625, -3.5, -3.375, -3.25, 
    -3.125, -3.0, -2.875, -2.75, -2.625, -2.5, -2.375, -2.25, -2.125, -2.0,
    -1.9375, -1.875, -1.8125, -1.75, -1.6875, -1.625, -1.5625, -1.5, -1.4375, 
    -1.375, -1.3125, -1.25, -1.1875, -1.125, -1.0625, -1.0, -0.96875, -0.9375,
    -0.90625, -0.875, -0.84375, -0.8125, -0.78125, -0.75, -0.71875, -0.6875, 
    -0.65625, -0.625, -0.59375, -0.5625, -0.53125, -0.5, -0.484375, -0.46875,
    -0.453125, -0.4375, -0.421875, -0.40625, -0.390625, -0.375, -0.359375, 
    -0.34375, -0.328125, -0.3125, -0.296875, -0.28125, -0.265625, -0.25, -0.2421875,
    -0.234375, -0.2265625, -0.21875, -0.2109375, -0.203125, -0.1953125, 
    -0.1875, -0.1796875, -0.171875, -0.1640625, -0.15625, -0.1484375, -0.140625,
    -0.1328125, 0.0,
    0.1328125, 0.140625, 0.1484375, 0.15625, 0.1640625, 0.171875, 0.1796875, 0.1875, 0.1953125, 0.203125, 0.2109375, 0.21875, 0.2265625, 0.234375, 0.2421875,
    0.25, 0.265625, 0.28125, 0.296875, 0.3125, 0.328125, 0.34375, 0.359375, 0.375, 0.390625, 0.40625, 0.421875, 0.4375, 0.453125, 0.46875, 0.484375, 0.5,
    0.53125, 0.5625, 0.59375, 0.625, 0.65625, 0.6875, 0.71875, 0.75, 0.78125, 0.8125, 0.84375, 0.875, 0.90625, 0.9375, 0.96875, 1.0, 1.0625, 1.125, 1.1875,
    1.25, 1.3125, 1.375, 1.4375, 1.5, 1.5625, 1.625, 1.6875, 1.75, 1.8125, 1.875, 1.9375, 2.0, 2.125, 2.25, 2.375, 2.5, 2.625, 2.75, 2.875, 3.0, 3.125, 3.25,
    3.375, 3.5, 3.625, 3.75, 3.875, 4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 5.75, 6.0, 6.25, 6.5, 6.75, 7.0, 7.25, 7.5, 7.75, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0,
    11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0
]
FP8_E4M3 = [
    -480.0, -448.0, -416.0, -384.0, -352.0, -320.0, -288.0, -256.0, -240.0, 
    -224.0, -208.0, -192.0, -176.0, -160.0, -144.0, -128.0, -120.0, -112.0,
    -104.0, -96.0, -88.0, -80.0, -72.0, -64.0, -60.0, -56.0, -52.0, -48.0, 
    -44.0, -40.0, -36.0, -32.0, -30.0, -28.0, -26.0, -24.0, -22.0, -20.0, -18.0,
    -16.0, -15.0, -14.0, -13.0, -12.0, -11.0, -10.0, -9.0, -8.0, -7.5, -7.0, 
    -6.5, -6.0, -5.5, -5.0, -4.5, -4.0, -3.75, -3.5, -3.25, -3.0, -2.75, -2.5,
    -2.25, -2.0, -1.875, -1.75, -1.625, -1.5, -1.375, -1.25, -1.125, -1.0, 
    -0.9375, -0.875, -0.8125, -0.75, -0.6875, -0.625, -0.5625, -0.5, -0.46875,
    -0.4375, -0.40625, -0.375, -0.34375, -0.3125, -0.28125, -0.25, -0.234375, 
    -0.21875, -0.203125, -0.1875, -0.171875, -0.15625, -0.140625, -0.125,
    -0.1171875, -0.109375, -0.1015625, -0.09375, -0.0859375, -0.078125, 
    -0.0703125, -0.0625, -0.05859375, -0.0546875, -0.05078125, -0.046875, -0.04296875,
    -0.0390625, -0.03515625, -0.03125, -0.029296875, -0.02734375, -0.025390625, 
    -0.0234375, -0.021484375, -0.01953125, -0.017578125, -0.015625, -0.0146484375,
    -0.013671875, -0.0126953125, -0.01171875, 
    -0.0107421875, -0.009765625, -0.0087890625, 0.0,
    0.0087890625, 0.009765625, 0.0107421875, 0.01171875, 0.0126953125, 0.013671875, 0.0146484375, 0.015625, 0.017578125, 0.01953125, 0.021484375, 0.0234375,
    0.025390625, 0.02734375, 0.029296875, 0.03125, 0.03515625, 0.0390625, 0.04296875, 0.046875, 0.05078125, 0.0546875, 0.05859375, 0.0625, 0.0703125, 0.078125,
    0.0859375, 0.09375, 0.1015625, 0.109375, 0.1171875, 0.125, 0.140625, 0.15625, 0.171875, 0.1875, 0.203125, 0.21875, 0.234375, 0.25, 0.28125, 0.3125, 0.34375,
    0.375, 0.40625, 0.4375, 0.46875, 0.5, 0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375, 1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875, 2.0, 2.25, 2.5,
    2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0,
    30.0, 32.0, 36.0, 40.0, 44.0, 48.0, 52.0, 56.0, 60.0, 64.0, 72.0, 80.0, 88.0, 96.0, 104.0, 112.0, 120.0, 128.0, 144.0, 160.0, 176.0, 192.0, 208.0, 224.0,
    240.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, 480.0
]
FP8_E5M2 = [
    -114688.0, -98304.0, -81920.0, -65536.0, -57344.0, -49152.0, -40960.0, -32768.0, 
    -28672.0, -24576.0, -20480.0, -16384.0, -14336.0, -12288.0, -10240.0, -8192.0,
    -7168.0, -6144.0, -5120.0, -4096.0, -3584.0, -3072.0, -2560.0, -2048.0, 
    -1792.0, -1536.0, -1280.0, -1024.0, -896.0, 
    -768.0, -640.0, -512.0, -448.0, -384.0,
    -320.0, -256.0, -224.0, -192.0, -160.0, -128.0, -112.0, -96.0, -80.0, -64.0, 
    -56.0, -48.0, -40.0, -32.0, -28.0, -24.0, 
    -20.0, -16.0, -14.0, -12.0, -10.0, -8.0,
    -7.0, -6.0, -5.0, -4.0, -3.5, -3.0, -2.5, -2.0, -1.75, -1.5, -1.25, -1.0, -0.875, 
    -0.75, -0.625, -0.5, -0.4375, -0.375, -0.3125, -0.25, -0.21875, -0.1875, -0.15625,
    -0.125, -0.109375, -0.09375, -0.078125, -0.0625, -0.0546875, -0.046875, 
    -0.0390625, -0.03125, -0.02734375, -0.0234375, 
    -0.01953125, -0.015625, -0.013671875,
    -0.01171875, -0.009765625, -0.0078125, -0.0068359375, -0.005859375, -0.0048828125, 
    -0.00390625, -0.00341796875, -0.0029296875, -0.00244140625, -0.001953125,
    -0.001708984375, -0.00146484375, -0.001220703125, -0.0009765625, -0.0008544921875, 
    -0.000732421875, -0.0006103515625, -0.00048828125, 
    -0.00042724609375, -0.0003662109375,
    -0.00030517578125, -0.000244140625, -0.000213623046875, -0.00018310546875, -0.000152587890625, -0.0001220703125, -0.0001068115234375, 
    -9.1552734375e-05, -7.62939453125e-05,
    -6.103515625e-05, -5.340576171875e-05, -4.57763671875e-05, -3.814697265625e-05, 0.0,
    3.814697265625e-05, 4.57763671875e-05, 5.340576171875e-05, 6.103515625e-05, 7.62939453125e-05, 9.1552734375e-05, 0.0001068115234375, 0.0001220703125, 0.000152587890625,
    0.00018310546875, 0.000213623046875, 0.000244140625, 0.00030517578125, 0.0003662109375, 0.00042724609375, 0.00048828125, 0.0006103515625, 0.000732421875, 0.0008544921875,
    0.0009765625, 0.001220703125, 0.00146484375, 0.001708984375, 0.001953125, 0.00244140625, 0.0029296875, 0.00341796875, 0.00390625, 0.0048828125, 0.005859375, 0.0068359375,
    0.0078125, 0.009765625, 0.01171875, 0.013671875, 0.015625, 0.01953125, 0.0234375, 0.02734375, 0.03125, 0.0390625, 0.046875, 0.0546875, 0.0625, 0.078125, 0.09375, 0.109375,
    0.125, 0.15625, 0.1875, 0.21875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0,
    16.0, 20.0, 24.0, 28.0, 32.0, 40.0, 48.0, 56.0, 64.0, 80.0, 96.0, 112.0, 128.0, 160.0, 192.0, 224.0, 256.0, 320.0, 384.0, 448.0, 512.0, 640.0, 768.0, 896.0, 1024.0, 1280.0,
    1536.0, 1792.0, 2048.0, 2560.0, 3072.0, 3584.0, 4096.0, 5120.0, 6144.0, 7168.0, 8192.0, 10240.0, 12288.0, 14336.0, 16384.0, 20480.0, 24576.0, 28672.0, 32768.0, 40960.0, 49152.0,
    57344.0, 65536.0, 81920.0, 98304.0, 114688.0
]

# --------------------------------------------------------------------------------------------
DATATYPE_MAPPING_3_BIT = {
    'int3': INT3, 'fp3': FP3,
    'fp3_er_pos': FP3_ER_POS, 'fp3_er_neg': FP3_ER_NEG,
    'fp3_ea_pos': FP3_EA_POS, 'fp3_ea_neg': FP3_EA_NEG,
}
DATATYPE_MAPPING_3_BIT_MX = {
    'mx_int3': INT3, 'mx_fp3': FP3
}

DATATYPE_MAPPING_4_BIT = {
    'int4': INT4, 'fp4': FP4_E2M1, 'flint4': FLINT4,
    'fp4_er_pos': FP4_ER_POS, 'fp4_er_neg': FP4_ER_NEG,
    'fp4_ea_pos': FP4_EA_POS, 'fp4_ea_neg': FP4_EA_NEG,
}
DATATYPE_MAPPING_4_BIT_MX = {
    'mx_int4': INT4, 'mx_fp4': FP4_E2M1
}

DATATYPE_MAPPING_5_BIT = {
    'int5': INT5, 'fp5': FP5_E2M2, 'flint5': FLINT5,
    'fp5_e2m2': FP5_E2M2, 'fp5_e3m1': FP5_E3M1
}

DATATYPE_MAPPING_6_BIT = {
    'int6': INT6, 'fp6': FP6_E2M3,
    'fp6_e2m3': FP6_E2M3, 'fp6_e3m2': FP6_E3M2
}
# ---------- bit-width extension ------------------------"Inseo"
DATATYPE_MAPPING_8_BIT = {'int8': INT8,
                          'fp8_e2m5': FP8_E2M5, 'fp8_e3m4': FP8_E3M4,
                          'fp8_e4m3': FP8_E4M3, 'fp8_e5m2': FP8_E5M2
                          }
# -------------------------------------------------------------


""" @torch.no_grad()
def quant_int(w_fp16, wq_bits:int=4, group_size: Optional[int]=None):  
    if (group_size is None) or (group_size <= 0):
        w_fp16_new = w_fp16.to(torch.float16)
    else:
        K, C = w_fp16.size() # output channel, input channel
        NUM_GROUP = C // group_size
        w_fp16_new = w_fp16.unsqueeze(-1).reshape(K, NUM_GROUP, group_size).to(torch.float16)
    
    rmax = torch.amax(w_fp16_new.abs(), dim=-1, keepdim=True)
    qmax = 2 ** (wq_bits - 1) - 1
    qmin = -qmax
    scale_fp = rmax / qmax
    scale_fp = scale_fp.clamp(min=1e-5, max=1e4)
    q_tensor = torch.clamp(torch.round(w_fp16_new / scale_fp), min=qmin, max=qmax)

    w_fp16_new = q_tensor * scale_fp
    if (group_size is None) or (group_size <= 0):
        return w_fp16_new
    else:
        return w_fp16_new.reshape(K, C) """

@torch.no_grad()
def quant_int(w_fp16, wq_bits: int = 4, group_size: Optional[int] = None):
    """
        Symmetric INT quantization.
        group_size:
          -1        :per-tensor  
           0/None   :per-channel
           >0       :per-group
    """
    qmax = 2 ** (wq_bits-1)-1
    qmin = -qmax

    if group_size == -1:  # per -tensor
        w_fp16_new = w_fp16.to(torch.float16)
        rmax = w_fp16_new.abs().amax()
        scale_fp = (rmax/qmax).clamp(min=1e-5, max=1e4)
        q_tensor = torch.clamp(torch.round(
            w_fp16_new / scale_fp), min=qmin, max=qmax)
        return (q_tensor*scale_fp)

    if (group_size is None) or (group_size <= 0):  # per-channel
        w_fp16_new = w_fp16.to(torch.float16)
    else:
        K, C = w_fp16.size()                      # per-group
        NUM_GROUP = C // group_size
        w_fp16_new = w_fp16.unsqueeze(-1).reshape(K,
                                                  NUM_GROUP, group_size).to(torch.float16)

    rmax = torch.amax(w_fp16_new.abs(), dim=-1, keepdim=True)
    scale_fp = rmax / qmax
    scale_fp = scale_fp.clamp(min=1e-5, max=1e4)
    q_tensor = torch.clamp(torch.round(
        w_fp16_new / scale_fp), min=qmin, max=qmax)
    deq_tensor = q_tensor * scale_fp
    if (group_size is None) or (group_size <= 0):
        return deq_tensor
    else:
        return deq_tensor.reshape(K, C)


""" @torch.no_grad()
def quant_int_asym(w_fp16, wq_bits:int=4, group_size: Optional[int]=None):
    #
    #   Asymmetric INT quantization.
    #    
    if (group_size is None) or (group_size <= 0):
        w_fp16_new = w_fp16.to(torch.float16)
    else:
        K, C = w_fp16.size() # output channel, input channel
        NUM_GROUP = C // group_size
        w_fp16_new = w_fp16.unsqueeze(-1).reshape(K, NUM_GROUP, group_size).to(torch.float16)
    
    rmin = torch.amin(w_fp16_new, dim=-1, keepdim=True)
    rmax = torch.amax(w_fp16_new, dim=-1, keepdim=True)
    qmin = 0
    qmax = 2**wq_bits - 1
    scale_fp = (rmax - rmin) / (qmax - qmin)
    scale_fp = scale_fp.clamp(min=1e-5, max=1e4)
    zeropoint = torch.round(-rmin / scale_fp).clamp(min=qmin, max=qmax)

    q_tensor = torch.clamp(torch.round(w_fp16_new / scale_fp) + zeropoint, min=qmin, max=qmax)

    w_fp16_new = (q_tensor - zeropoint) * scale_fp
    if (group_size is None) or (group_size <= 0):
        return w_fp16_new
    else:
        return w_fp16_new.reshape(K, C)
 """

@torch.no_grad()
def quant_int_asym(w_fp16, wq_bits: int = 4, group_size: Optional[int] = None):
    """
        Asymmetric INT quantization.
        group_size:
          -1        :per-tensor  
           0/None   :per-channel
           >0       :per-group
    """
    qmin, qmax = 0, 2**wq_bits - 1

    if group_size == -1:  # per-tensor
        w_fp16_new = w_fp16.to(torch.float16)
        rmin = w_fp16_new.amin()
        rmax = w_fp16_new.amax()
        scale_fp = ((rmax - rmin) / (qmax - qmin)).clamp(min=1e-5, max=1e4)
        zp = torch.round(-rmin / scale_fp).clamp(min=qmin, max=qmax)
        q_tensor = torch.clamp(torch.round(
            w_fp16_new / scale_fp) + zp, min=qmin, max=qmax)
        return (q_tensor - zp)*scale_fp

    if (group_size is None) or (group_size <= 0):
        w_fp16_new = w_fp16.to(torch.float16)
    else:
        K, C = w_fp16.size()
        Ng = C // group_size
        w_fp16_new = w_fp16.unsqueeze(-1).reshape(K, Ng, group_size).to(torch.float16)

    rmin = torch.amin(w_fp16_new, dim=-1, keepdim=True)
    rmax = torch.amax(w_fp16_new, dim=-1, keepdim=True)
    scale_fp = ((rmax - rmin) / (qmax - qmin)).clamp(min=1e-5, max=1e4)
    zp = torch.round(-rmin / scale_fp).clamp(min=qmin, max=qmax)
    q_tensor = torch.clamp(torch.round(w_fp16_new / scale_fp) + zp, min=qmin, max=qmax)
    deq_tensor = (q_tensor - zp) * scale_fp
    if (group_size is None) or (group_size <= 0):
        return deq_tensor
    else:
        return deq_tensor.reshape(K, C)

@torch.no_grad()
def quant_mx(w_fp16, wq_bits: int = 4, datatype: str = "", group_size: int = 32):

    #   MX quantization.
    #   Reference: https://github.com/microsoft/microxcaling/blob/7bc41952de394f5cc5e782baf132e7c7542eb4e4/mx/mx_ops.py

    if wq_bits == 3:
        DATATYPE_MAPPING = DATATYPE_MAPPING_3_BIT_MX
    elif wq_bits == 4:
        DATATYPE_MAPPING = DATATYPE_MAPPING_4_BIT_MX
    else:
        raise ValueError(
            f"Currently only support 3-bit, 4-bit quantization, not {wq_bits}-bit")

    assert datatype in DATATYPE_MAPPING, f"unexpected data type {datatype}."

    allow_value = DATATYPE_MAPPING[datatype]
    mid_value = [(allow_value[i] + allow_value[i + 1]) /
                 2 for i in range(len(allow_value) - 1)]
    K, C = w_fp16.size()  # output channel, input channel
    NUM_GROUP = C // group_size
    w_fp16_new = w_fp16.unsqueeze(-1).reshape(K,
                                              NUM_GROUP, group_size).to(torch.float32)

    shared_exp, _ = torch.max(w_fp16_new.abs(), dim=-1, keepdim=True)
    shared_exp = torch.floor(torch.log2(shared_exp))
    w_fp16_new = w_fp16_new / (2**shared_exp)
    qmax = max([abs(x) for x in allow_value])
    scale = 1 / (qmax / 2)
    x = w_fp16_new / scale

    q_tensor = torch.zeros_like(x)
    for i in range(len(allow_value)):
        data = allow_value[i]
        if i == 0:
            q_tensor += torch.where(x <= mid_value[i], data, 0)
        elif i == len(allow_value) - 1:
            q_tensor += torch.where(x > mid_value[i - 1], data, 0)
        else:
            q_tensor += torch.where((mid_value[i - 1] < x)
                                    & (x <= mid_value[i]), data, 0)

    w_fp16_new = q_tensor * scale * (2**shared_exp)
    return w_fp16_new.reshape(K, C).to(torch.float16)


""" #--------- Basic FP quant -----------------
@torch.no_grad()
def quant_fp(w_fp16, wq_bits:int=4, datatype: str="", group_size: int=32):
    if wq_bits == 3:
        DATATYPE_MAPPING = DATATYPE_MAPPING_3_BIT
    elif wq_bits == 4:
        DATATYPE_MAPPING = DATATYPE_MAPPING_4_BIT
    elif wq_bits == 8:
        DATATYPE_MAPPING = DATATYPE_MAPPING_8_BIT
    else:
        raise ValueError(f"Currently only support 3-bit, 4-bit, and 8-bit quantizaton, not {wq_bits}-bit")
    
    assert datatype in DATATYPE_MAPPING, f"unexpected data type {datatype}."
    
    allow_value = DATATYPE_MAPPING[datatype]
    mid_value = [(allow_value[i] + allow_value[i + 1]) / 2 for i in range(len(allow_value) - 1)]
    K, C = w_fp16.size() # output channel, input channel
    NUM_GROUP = C // group_size
    w_fp16_new = w_fp16.unsqueeze(-1).reshape(K, NUM_GROUP, group_size).to(torch.float32)
    
    max_abs = w_fp16_new.abs().amax(dim=-1, keepdim=True)
    qmax = max([abs(x) for x in allow_value])
    scale = max_abs / qmax
    
    x = w_fp16_new / scale
    
    q_tensor = torch.zeros_like(x)
    for i in range(len(allow_value)):
        data = allow_value[i]
        if i == 0:
            q_tensor += torch.where(x <= mid_value[i], data, 0)
        elif i == len(allow_value) - 1:
            q_tensor += torch.where(x > mid_value[i - 1], data, 0)
        else:
            q_tensor += torch.where((mid_value[i - 1] < x) & (x <= mid_value[i]), data, 0)
    
    w_fp16_new = q_tensor * scale
    return w_fp16_new.reshape(K, C).to(torch.float16) """

# ---------------------------


""" @torch.no_grad()
def quant_datatype(w_fp16, wq_bits: int = 4, datatype: str = "", group_size: Optional[int] = None):
    if datatype == "fp8_e5m2":
        from torch import tensor
        w = w_fp16.to(torch.float32)
        if group_size and group_size > 0:
            K, C = w.size()
            w = w.unsqueeze(-1).reshape(K, C // group_size, group_size)
        # scale 계산
        rmax = torch.amax(w.abs(), dim=-1, keepdim=True)
        allow = tensor(
            DATATYPE_MAPPING_8_BIT[datatype], dtype=torch.float32, device=w.device)
        qmax = allow.abs().amax()
        scale = (rmax / qmax).clamp(min=1e-5, max=1e4)
        # quantize
        x = w / scale
        mid = (allow[:-1] + allow[1:]) * 0.5
        q = torch.zeros_like(x, dtype=torch.float32)
        edges = torch.cat([tensor([-float('inf')], device=w.device),
                          mid, tensor([float('inf')], device=w.device)])
        for i, v in enumerate(allow):
            mask = (edges[i] < x) & (x <= edges[i+1])
            q = torch.where(mask, v, q)
        w_deq = q * scale
        return (w_deq.reshape_as(w_fp16)
                if group_size and group_size > 0 else w_deq)
    if wq_bits == 3:
        DATATYPE_MAPPING = DATATYPE_MAPPING_3_BIT
    elif wq_bits == 4:
        DATATYPE_MAPPING = DATATYPE_MAPPING_4_BIT
    elif wq_bits == 5:
        DATATYPE_MAPPING = DATATYPE_MAPPING_5_BIT
    elif wq_bits == 6:
        DATATYPE_MAPPING = DATATYPE_MAPPING_6_BIT
    elif wq_bits == 8:                              # inseo
        DATATYPE_MAPPING = DATATYPE_MAPPING_8_BIT
    else:
        raise ValueError(
            f"Currently only support 3-, 4-, 5-,6- and 8-bit quantization, not {wq_bits}-bit")

    assert datatype in DATATYPE_MAPPING, f"unexpected data type {datatype}."

    allow_value = DATATYPE_MAPPING[datatype]
    mid_value = [(allow_value[i] + allow_value[i + 1]) /
                 2 for i in range(len(allow_value) - 1)]

    if (group_size is None) or (group_size <= 0):
        w_fp16_new = w_fp16.to(torch.float16)
    else:
        K, C = w_fp16.size()  # output channel, input channel
        NUM_GROUP = C // group_size
        w_fp16_new = w_fp16.unsqueeze(-1).reshape(K,
                                                  NUM_GROUP, group_size).to(torch.float16)

    rmax = torch.amax(w_fp16_new.abs(), dim=-1, keepdim=True)
    qmax = max([abs(x) for x in allow_value])
    scale_fp = rmax / qmax
    scale_fp = scale_fp.clamp(min=1e-5, max=1e4)
    x = w_fp16_new / scale_fp

    q_tensor = torch.zeros_like(x)
    for i in range(len(allow_value)):
        data = allow_value[i]
        if i == 0:
            q_tensor += torch.where(x <= mid_value[i], data, 0)
        elif i == len(allow_value) - 1:
            q_tensor += torch.where(x > mid_value[i - 1], data, 0)
        else:
            q_tensor += torch.where((mid_value[i - 1] < x)
                                    & (x <= mid_value[i]), data, 0)

    w_fp16_new = q_tensor * scale_fp

    if (group_size is None) or (group_size <= 0):
        return w_fp16_new
    else:
        return w_fp16_new.reshape(K, C) """
        
@torch.no_grad()
def quant_datatype(w_fp16, wq_bits: int = 4, datatype: str = "", group_size: Optional[int] = None):
    if datatype == "fp8_e5m2":
        from torch import tensor
        w = w_fp16.to(torch.float32)
        
        if group_size == -1:  # per-tensor
            rmax  = w.abs().amax()
            allow = tensor(DATATYPE_MAPPING_8_BIT[datatype], dtype=torch.float32, device=w.device)
            qmax  = allow.abs().amax()
            scale = (rmax / qmax).clamp(min=1e-5, max=1e4)
            x   = w / scale
            mid = (allow[:-1] + allow[1:]) * 0.5
            q   = torch.zeros_like(x, dtype=torch.float32)
            edges = torch.cat([tensor([-float('inf')], device=w.device), mid, tensor([float('inf')], device=w.device)])
            for i, v in enumerate(allow):
                mask = (edges[i] < x) & (x <= edges[i+1])
                q = torch.where(mask, v, q)
            w_deq = q * scale
            return w_deq  

        # per-channel/per-group 
        if group_size and group_size > 0:
            K, C = w.size()
            w = w.unsqueeze(-1).reshape(K, C // group_size, group_size)
        rmax   = torch.amax(w.abs(), dim=-1, keepdim=True)
        allow  = tensor(DATATYPE_MAPPING_8_BIT[datatype], dtype=torch.float32, device=w.device)
        qmax   = allow.abs().amax()
        scale  = (rmax / qmax).clamp(min=1e-5, max=1e4)
        x      = w / scale
        mid    = (allow[:-1] + allow[1:]) * 0.5
        q      = torch.zeros_like(x, dtype=torch.float32)
        edges  = torch.cat([tensor([-float('inf')], device=w.device), mid, tensor([float('inf')], device=w.device)])
        for i, v in enumerate(allow):
            mask = (edges[i] < x) & (x <= edges[i+1])
            q = torch.where(mask, v, q)
        w_deq = q * scale
        return (w_deq.reshape_as(w_fp16) if group_size and group_size > 0 else w_deq)
    
    if wq_bits == 3:
        DATATYPE_MAPPING = DATATYPE_MAPPING_3_BIT
    elif wq_bits == 4:
        DATATYPE_MAPPING = DATATYPE_MAPPING_4_BIT
    elif wq_bits == 5:
        DATATYPE_MAPPING = DATATYPE_MAPPING_5_BIT
    elif wq_bits == 6:
        DATATYPE_MAPPING = DATATYPE_MAPPING_6_BIT
    elif wq_bits == 8:                              # inseo
        DATATYPE_MAPPING = DATATYPE_MAPPING_8_BIT
    else:
        raise ValueError(
            f"Currently only support 3-, 4-, 5-,6- and 8-bit quantization, not {wq_bits}-bit")

    assert datatype in DATATYPE_MAPPING, f"unexpected data type {datatype}."

    allow_value = DATATYPE_MAPPING[datatype]
    mid_value = [(allow_value[i] + allow_value[i + 1]) /
                 2 for i in range(len(allow_value) - 1)]

    if group_size == -1:  # per-tensor
        w32 = w_fp16.to(torch.float32)
        rmax = w32.abs().amax()
        qmax = max(abs(x) for x in allow_value)
        scale = (rmax / qmax).clamp(min=1e-5, max=1e4)
        x = w32 / scale

        q = torch.zeros_like(x)
        for i in range(len(allow_value)):
            data = allow_value[i]
            if i == 0:                  q += torch.where(x <= mid_value[i], data, 0)
            elif i == len(allow_value)-1: q += torch.where(x >  mid_value[i-1], data, 0)
            else:                       q += torch.where((mid_value[i-1] < x) & (x <= mid_value[i]), data, 0)
        return (q * scale).to(torch.float16)

    # per-channel/per-group 
    if (group_size is None) or (group_size <= 0):
        w_view = w_fp16.to(torch.float16)
    else:
        K, C = w_fp16.size()
        Ng = C // group_size
        w_view = w_fp16.unsqueeze(-1).reshape(K, Ng, group_size).to(torch.float16)

    rmax = torch.amax(w_view.abs(), dim=-1, keepdim=True)
    qmax = max([abs(x) for x in allow_value])
    scale_fp = (rmax / qmax).clamp(min=1e-5, max=1e4)
    x = w_view / scale_fp

    q_tensor = torch.zeros_like(x)
    for i in range(len(allow_value)):
        data = allow_value[i]
        if i == 0:                  q_tensor += torch.where(x <= mid_value[i], data, 0)
        elif i == len(allow_value)-1: q_tensor += torch.where(x >  mid_value[i-1], data, 0)
        else:                       q_tensor += torch.where((mid_value[i-1] < x) & (x <= mid_value[i]), data, 0)

    deq = q_tensor * scale_fp
    if (group_size is None) or (group_size <= 0):
        return deq
    else:
        return deq.reshape(K, C)


@torch.no_grad()
def search_datatype(w_fp16, wq_bits: int = 4, datatype: str = 'mixed_bitmod', group_size: Optional[int] = None):
    if wq_bits == 3:
        if datatype == 'mixed_bitmod':
            datatype_list = ['fp3_er_pos', 'fp3_er_neg',
                             'fp3_ea_pos', 'fp3_ea_neg']
        elif datatype == 'mixed_er':
            datatype_list = ['fp3_er_pos', 'fp3_er_neg']
        elif datatype == 'mixed_ea':
            datatype_list = ['fp3_ea_pos', 'fp3_ea_neg']
        elif datatype == 'mixed_ant':
            datatype_list = ['int3', 'fp3']
    elif wq_bits == 4:
        if datatype == 'mixed_bitmod':
            datatype_list = ['fp4_er_pos', 'fp4_er_neg',
                             'fp4_ea_pos', 'fp4_ea_neg']
        elif datatype == 'mixed_er':
            datatype_list = ['fp4_er_pos', 'fp4_er_neg']
        elif datatype == 'mixed_ea':
            datatype_list = ['fp4_ea_pos', 'fp4_ea_neg']
        elif datatype == 'mixed_ant':
            datatype_list = ['int4', 'flint4']
    else:
        raise ValueError(
            f"Currently only support 3-bit and 4-bit mixed quantization, not {wq_bits}-bit")

    K, C = w_fp16.size()  # output channel, input channel
    if (group_size is None) or (group_size <= 0):
        group_size = C
    NUM_GROUP = C // group_size
    w_fp16 = w_fp16.unsqueeze(-1).reshape(K, NUM_GROUP, group_size)
    q_tensor = torch.zeros_like(w_fp16)

    error = torch.full([K, NUM_GROUP], 1e3,
                       dtype=w_fp16.dtype, device=w_fp16.device)
    for datatype in datatype_list:
        w_fp16_tmp = quant_datatype(
            w_fp16, wq_bits=wq_bits, datatype=datatype, group_size=None)
        quant_error = (w_fp16_tmp - w_fp16).pow(2).mean(-1)
        update_mask = torch.lt(quant_error, error)
        error[update_mask] = quant_error[update_mask]
        q_tensor[update_mask] = w_fp16_tmp[update_mask]

        del w_fp16_tmp, quant_error, update_mask

    return q_tensor.reshape(K, C)


def quant_model(model, wq_bits: Optional[int] = None, wq_datatype: Optional[str] = None, wq_groupsize: Optional[int] = None):
    if (wq_datatype is None) or (wq_datatype in ["fp16", "fp32"]):
        print("Not applying quantization")
        time.sleep(2)
    elif (wq_datatype.startswith("int")) and ("asym" in wq_datatype):
        print(
            f"Applying asymmetric INT quantization with bits: {wq_bits}, group size: {wq_groupsize}")
        time.sleep(2)
        for n, m in model.named_modules():
            if isinstance(m, torch.nn.Linear):
                print(f'Quantizing layer: {n}')
                m.weight.data = quant_int_asym(
                    m.weight.data, wq_bits=wq_bits, group_size=wq_groupsize)
    elif (wq_datatype.startswith("int")) and ("asym" not in wq_datatype):
        print(
            f"Applying symmetric INT quantization with bits: {wq_bits}, group size: {wq_groupsize}")
        time.sleep(2)
        for n, m in model.named_modules():
            if isinstance(m, torch.nn.Linear):
                print(f'Quantizing layer: {n}')
                m.weight.data = quant_int(
                    m.weight.data, wq_bits=wq_bits, group_size=wq_groupsize)
    elif ("mx" in wq_datatype):
        '''
            We use hard-coded group size 32 based on the Open Compute Standard
            https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
        '''
        print(
            f"Applying MX quantization with bits: {wq_bits}, datatype: {wq_datatype}, group size: 32")
        time.sleep(2)
        for n, m in model.named_modules():
            if isinstance(m, torch.nn.Linear):
                print(f'Quantizing layer: {n}')
                m.weight.data = quant_mx(
                    m.weight.data, wq_bits=wq_bits, datatype=wq_datatype, group_size=32)
    elif ("mixed" in wq_datatype):
        print(
            f"Applying mixed datatype quantization with bits: {wq_bits}, datatype: {wq_datatype}, group size: {wq_groupsize}")
        time.sleep(2)
        for n, m in model.named_modules():
            if isinstance(m, torch.nn.Linear):
                print(f'Quantizing layer: {n}')
                m.weight.data = search_datatype(
                    m.weight.data, wq_bits=wq_bits, datatype=wq_datatype, group_size=wq_groupsize)
    elif ("fp" in wq_datatype):
        print(
            f"Applying floating-point datatype quantization with bits: {wq_bits}, group size: {wq_groupsize}")
        time.sleep(2)
        for n, m in model.named_modules():
            if isinstance(m, torch.nn.Linear):
                print(f'Quantizing layer: {n}')
                m.weight.data = quant_datatype(
                    m.weight.data, wq_bits=wq_bits, datatype=wq_datatype, group_size=wq_groupsize)
    else:
        raise ValueError(f"Unsupported datatype {wq_datatype}")


''' Inseo '''
# =======================================================================
#  Add quantized tensor value to onnx.data
# =======================================================================


# ---------- 1)  INT/FP quantization + metadata ------------------------------------------------

@torch.no_grad()
def _int_quant_meta(w_fp16: torch.Tensor,
                    wq_bits: int = 4,
                    asym: bool = False,
                    group_size: Optional[int] = None):
    """
    return
    • de-quant FP16  
    • q_tensor[int8] (=-7‥7 / 0‥15)
    simultaneously
    """
    if group_size == -1:  # per-tensor
        w_view = w_fp16.to(torch.float16)
        if asym:
            qmin, qmax = 0, 2**wq_bits - 1
            rmin = w_view.amin()
            rmax = w_view.amax()
            scale = ((rmax - rmin) / (qmax - qmin)).clamp(min=1e-5, max=1e4)
            zp = torch.round(-rmin / scale).clamp(min=qmin, max=qmax)
            q = torch.clamp(torch.round(w_view / scale) + zp, min=qmin, max=qmax)
            w_deq = (q - zp) * scale
            return w_deq.reshape_as(w_fp16), q.to(torch.int8), scale, zp
        else:
            qmax = 2 ** (wq_bits - 1) - 1
            rmax = w_view.abs().amax()
            scale = (rmax / qmax).clamp_(min=1e-5, max=1e4)
            q = torch.round(w_view / scale).clamp_(min=-qmax, max=qmax)
            w_deq = q * scale
            return w_deq.reshape_as(w_fp16), q.to(torch.uint8), scale, None # unit8 반영 안된채로 export되어있음 (int8로 export됨)
    
    # ---- per-channel·per-group reshape ---------------------------------
    if (group_size is None) or (group_size <= 0):
        w_view = w_fp16.to(torch.float16)
    else:                                   # [K,C] → [K,Ng,gs]
        K, C = w_fp16.size()
        Ng = C // group_size
        w_view = w_fp16.unsqueeze(-1).reshape(K, Ng,
                                              group_size).to(torch.float16)

    if asym:                                # ---------- Asymmetric -------
        rmin = torch.amin(w_view, dim=-1, keepdim=True)
        rmax = torch.amax(w_view, dim=-1, keepdim=True)
        qmin, qmax = 0, 2 ** wq_bits - 1
        scale = (rmax - rmin) / (qmax - qmin)
        scale = scale.clamp(min=1e-5, max=1e4)
        zp = torch.round(-rmin / scale).clamp(min=qmin, max=qmax)
        q = torch.round(w_view / scale) + zp
        q = torch.clamp(q, min=qmin, max=qmax)
        w_deq = (q - zp) * scale
        return w_deq.reshape_as(w_fp16), q.to(torch.uint8), scale.squeeze(-1), zp.squeeze(-1)

    else:                                   # ---------- Symmetric --------
        rmax = torch.amax(w_view.abs(), dim=-1, keepdim=True)
        qmax = 2 ** (wq_bits - 1) - 1
        scale = (rmax / qmax).clamp_(min=1e-5, max=1e4)
        q = torch.round(w_view / scale).clamp_(
            min=-qmax, max=qmax)            # -7‥+7(4bit) / -127‥+127(8bit)
        w_deq = q * scale
        return w_deq.reshape_as(w_fp16), q.to(torch.int8), scale.squeeze(-1), None

# ----------  FP-quantization + metadata  ---------------------------------
@torch.no_grad()
def _fp_quant_meta(w_fp16, wq_bits:int=4, datatype: str="", group_size: Optional[int]=None):
    if wq_bits == 3:
        DATATYPE_MAPPING = DATATYPE_MAPPING_3_BIT
    elif wq_bits == 4:
        DATATYPE_MAPPING = DATATYPE_MAPPING_4_BIT
    elif wq_bits == 5:
        DATATYPE_MAPPING = DATATYPE_MAPPING_5_BIT
    elif wq_bits == 6:
        DATATYPE_MAPPING = DATATYPE_MAPPING_6_BIT
    elif wq_bits == 8:
        DATATYPE_MAPPING = DATATYPE_MAPPING_8_BIT
    else:
        raise ValueError(f"Currently only support 3-, 4-, 5-,6- and 8-bit quantization, not {wq_bits}-bit")

    assert datatype in DATATYPE_MAPPING, f"unexpected data type {datatype}."

    # ===== FP8_E5M2:FP32 =====
    if datatype == "fp8_e5m2":
        w = w_fp16.to(torch.float32)

        # per-group: [K,C] -> [K,Ng,gs]
        if group_size and group_size > 0:
            K, C = w.size()
            Ng = C // group_size
            w = w.unsqueeze(-1).reshape(K, Ng, group_size)

        allow = torch.tensor(DATATYPE_MAPPING_8_BIT[datatype],
                             dtype=torch.float32, device=w.device)
        qmax  = allow.abs().amax()

        # per-tensor는 스칼라 scale, 그 외엔 마지막 축 기준
        rmax = w.abs().amax() if group_size == -1 else torch.amax(w.abs(), dim=-1, keepdim=True)
        scale = (rmax / qmax).clamp(min=1e-5, max=1e4)
        x = w / scale

        mid   = (allow[:-1] + allow[1:]) * 0.5
        q     = torch.zeros_like(x, dtype=torch.float32)
        edges = torch.cat([torch.tensor([-float('inf')], device=w.device),
                           mid, torch.tensor([float('inf')], device=w.device)])
        for i, v in enumerate(allow):
            mask = (edges[i] < x) & (x <= edges[i+1])
            q = torch.where(mask, v, q)

        w_deq = q * scale
        if group_size and group_size > 0:
            return w_deq.reshape(K, C), q.reshape(K, C), scale.squeeze(-1)
        else:
            return w_deq.reshape_as(w_fp16), q.reshape_as(w_fp16), (scale if scale.ndim == 0 else scale.squeeze(-1))

    # ===== 그 외 FP format: FP16 return =====
    allow_value = DATATYPE_MAPPING[datatype]
    mid_value = [(allow_value[i] + allow_value[i + 1]) / 2 for i in range(len(allow_value) - 1)]

    # --- per-tensor ---
    if group_size == -1:
        w16 = w_fp16.to(torch.float16)
        rmax = w16.abs().amax()
        qmax = max(abs(x) for x in allow_value)
        scale_fp = (rmax / qmax).clamp(min=1e-5, max=1e4)
        x = w16 / scale_fp

        q_tensor = torch.zeros_like(x)
        for i, data in enumerate(allow_value):
            if i == 0:                    q_tensor += torch.where(x <= mid_value[i], data, 0)
            elif i == len(allow_value)-1: q_tensor += torch.where(x >  mid_value[i-1], data, 0)
            else:                         q_tensor += torch.where((mid_value[i-1] < x) & (x <= mid_value[i]), data, 0)

        w_deq = q_tensor * scale_fp
        return w_deq.reshape_as(w_fp16), q_tensor.to(torch.float16), scale_fp  # scale: scalar

    # --- per-channel / per-group ---
    if (group_size is None) or (group_size <= 0):   # per-channel
        w_view = w_fp16.to(torch.float16)
    else:                                           # per-group: [K,C]→[K,Ng,gs]
        K, C = w_fp16.size()
        Ng = C // group_size
        w_view = w_fp16.unsqueeze(-1).reshape(K, Ng, group_size).to(torch.float16)

    rmax = torch.amax(w_view.abs(), dim=-1, keepdim=True)
    qmax = max(abs(x) for x in allow_value)
    scale_fp = (rmax / qmax).clamp(min=1e-5, max=1e4)
    x = w_view / scale_fp

    q_tensor = torch.zeros_like(x)
    for i, data in enumerate(allow_value):
        if i == 0:                    q_tensor += torch.where(x <= mid_value[i], data, 0)
        elif i == len(allow_value)-1: q_tensor += torch.where(x >  mid_value[i-1], data, 0)
        else:                         q_tensor += torch.where((mid_value[i-1] < x) & (x <= mid_value[i]), data, 0)

    w_deq = q_tensor * scale_fp
    if (group_size is None) or (group_size <= 0):
        return w_deq, q_tensor.to(torch.float16), scale_fp.squeeze(-1)       # scale: [K]
    else:
        return w_deq.reshape(K, C), q_tensor.to(torch.float16), scale_fp.squeeze(-1)  # scale: [K,Ng]



class _LinearDump(nn.Linear):
    def __init__(self, src: nn.Linear, wq_bits: int,
                 mode: str, group_size: Optional[int]):

        super().__init__(src.in_features, src.out_features,
                         bias=src.bias is not None)
        self.weight = nn.Parameter(src.weight.detach())
        if src.bias is not None:
            self.bias = nn.Parameter(src.bias.detach())

        if mode.startswith("int"):
            asym = "asym" in mode
            _, q, scale, zp = _int_quant_meta(
                self.weight.data, wq_bits=wq_bits,
                asym=asym, group_size=group_size)
        else:
            # e.g. fp4 / fp8_e2m5 / fp8_e3m4 ...
            _, q, scale = _fp_quant_meta(
                self.weight.data, wq_bits=wq_bits,
                datatype=mode, group_size=group_size)
            zp = None

        if mode == "fp8_e5m2":
            self.register_buffer("weight_q", q)                  # FP32
            self.register_buffer("weight_scale", scale)          # FP32
        else:
            self.register_buffer("weight_q", q.to(torch.float16))
            self.register_buffer("weight_scale", scale.to(torch.float16))
        if zp is not None:
            self.register_buffer("weight_zp", zp)

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        
        dummy  = self.weight_q.float().sum()
        dummy += self.weight_scale.float().sum()
        if hasattr(self, "weight_zp"):
            dummy += self.weight_zp.float().sum()
        return out + dummy * 0.0


# ---------- 3) original quant_model ➜ wrapping  -------------------------------
_original_quant_model = quant_model  # keep original


def quant_model(model, wq_bits=None, wq_datatype=None,
                wq_groupsize=None, _dump_int=True):

    _original_quant_model(model, wq_bits, wq_datatype, wq_groupsize)

    if not _dump_int or wq_datatype in (None, "fp16", "fp32"):
        return

    if not (wq_datatype.startswith("int") or wq_datatype.startswith("fp")):
        return  # 기타 datatype 무시

    for name, mod in list(model.named_modules()):
        if isinstance(mod, nn.Linear):
            parent, attr = _find_parent(model, name)
            setattr(parent, attr,
                    _LinearDump(mod, wq_bits, wq_datatype, wq_groupsize))


def _find_parent(root: nn.Module, target_name: str):
    parts = target_name.split('.')
    for p in parts[:-1]:
        root = getattr(root, p)
    return root, parts[-1]
