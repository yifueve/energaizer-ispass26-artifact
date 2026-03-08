
def prec_to_precision_bits(x):
    if (x == 'fp32'):
        return 32
    elif (x == 'fp16') or (x == 'bf16'):
        return 16
    elif (x == 'fp64'):
        return 64
    elif (x == 'int8'):
        return 8
    else:
        raise TypeError("ERROR: Precision {} for GEMM is currently not supported.".format(x))