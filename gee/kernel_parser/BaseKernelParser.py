import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('..')

class BaseKernelParser():
    def __init__(self, op, ops_supported):
        
        # Operation this parser supports: gemm_like, nonlinear
        self.op = op

        # Sub-operations this parser support
        # e.g., gemm_like: gemm, conv2d, fmha
        # e.g., nonlinear: softmax, layernorm, relu, gelu
        self.ops_supported = ops_supported

    def parse(self, query, op, **kwargs):
        raise NotImplementedError()
    
    def parse_dataframe(self, df, op, **kwargs):
        raise NotImplementedError()
    
