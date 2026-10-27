try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

def act_quant(x, block_size=128):
    raise NotImplementedError("FP8 quantization not used in BF16 mode")

def weight_dequant(x, s, block_size=128):
    raise NotImplementedError("FP8 dequantization not used in BF16 mode")

def fp8_gemm(a, b, scale_a, scale_b):
    raise NotImplementedError("FP8 gemm not used in BF16 mode")
