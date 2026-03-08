import os
import sys

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import yaml

sys.path.append('..')
from gee import get_gee
from gee.frontend_utils import prec_to_precision_bits

from estimator import Estimator

from tqdm import tqdm

class RooflineEstimator(Estimator):
    def __init__(self, gpu_config_yaml, lut_config_yaml, lut_folder_abs_path='/mnt/c/Users/KyungmiLee/Documents/gpu-energy-estimation/lut', **kwargs):
        super().__init__(gpu_config_yaml, lut_config_yaml, lut_folder_abs_path, **kwargs)

        # self.gee = get_gee(gpu_yaml_path=gpu_config_yaml, \
        #                    lut_yaml_path=lut_config_yaml, \
        #                    use_entire_references_for_estimate=False, \
        #                    dvfs_aware=False, \
        #                    lut_folder_abs_path=lut_folder_abs_path)
        
        with open(gpu_config_yaml, 'r') as f:
            self.gpu_config = yaml.safe_load(f)['gpu_configs']

    def _get_roofline_ridge(self, op_type, target_gpu_config=None):
        gpu_config = self.gpu_config if target_gpu_config is None else target_gpu_config

        gpu_hbm_bw = gpu_config['dram_bw'] # GB/s
        gpu_hbm_freq = gpu_config['dram_freq'] # MHz
        gpu_sm_flops = gpu_config['{}_{}_flops'.format(op_type[1], op_type[0])] # TFLOPs
        gpu_sm_freq = gpu_config['sm_max_freq'] # MHz
        gpu_hbm_byte_per_cycle = gpu_hbm_bw / (gpu_hbm_freq / 1000) # GByte / M * 1000 = Bytes/cycle 
        gpu_flop_per_cycle = gpu_sm_flops / (gpu_sm_freq / 1e6) # TFLOPs / M * 1e6 = FLOPs/cycle
        gpu_roofline_ridge = gpu_flop_per_cycle / gpu_hbm_byte_per_cycle

        return gpu_roofline_ridge

    def _get_elementwise_profile(self, query):
        bytes_per_precIn = prec_to_precision_bits(query['prec']) / 8
        
        if query['op'] == 'typecast_to_fp32':
            bytes_per_precOut = 4
        elif query['op'] == 'typecast_to_bf16':
            bytes_per_precOut = 2
        else:
            bytes_per_precOut = prec_to_precision_bits(query['prec']) / 8

        if query['op'] in ['pointwise_mul', 'pointwise_add', 'unspecified_tensor']:
            num_input_arg = 2
        else:
            num_input_arg = 1

        return bytes_per_precIn, bytes_per_precOut, num_input_arg

    def predict_single_kernel(self, query, op_type, target_freq, **kwargs):
        target_gpu_config = kwargs['target_gpu_config']

        if 'avg_freq' not in query.keys():
            query['avg_freq'] = target_freq

        if (op_type[0] == 'gemm') or (op_type[0] == 'conv2d'):
            query['byteM'] = prec_to_precision_bits(query['precM']) / 8
            query['byteA'] = prec_to_precision_bits(query['precA']) / 8
            query['mem_footprint'] = query['batch'] * (query['dimM'] * query['dimN'] * query['byteA'] + (query['dimM'] * query['dimK'] + query['dimN'] * query['dimK']) * query['byteM'])
            query['flop'] = query['batch'] * query['dimM'] * query['dimN'] * query['dimK'] * 2
            query['comp_intensity'] = query['flop'] / query['mem_footprint']
            # query['roofline_ridge'] = self.gee.gemm_estimator.gpu_roofline_ridge['{}_{}'.format(query['precM'], query['useTensorCore'])]
            query['roofline_ridge'] = self._get_roofline_ridge((query['precM'], 'tc' if query['useTensorCore'] else 'cuda'), target_gpu_config)
            query['bound'] = 'flops' if (query['comp_intensity'] > query['roofline_ridge']) else 'mem'
        
        elif (op_type[0] == 'softmax') or (op_type[0] == 'layernorm'):
            query['byte'] = prec_to_precision_bits(query['prec']) / 8
            query['mem_footprint'] = query['batch'] * query['dim'] * query['byte'] * 2

        elif (op_type[0] == 'elementwise'):
            # self.gee.elementwise_estimator.predict_kernel(query)
            # query = pd.DataFrame([query], index=[0])
            # self.gee.elementwise_estimator.analytical_model.model(query, train=False)
            bytes_per_precIn, bytes_per_precOut, num_input_arg = self._get_elementwise_profile(query)
            query['mem_footprint'] = query['dim'] * (num_input_arg * bytes_per_precIn + bytes_per_precOut)
            # query = query.iloc[0].to_dict()

        gpu_config = self.gpu_config if target_gpu_config is None else target_gpu_config
        if (op_type[0] == 'gemm') or (op_type[0] == 'conv2d'):
            math_bw = gpu_config['{}_{}_flops'.format('cuda' if query['useTensorCore'] == False else 'tc', query['precM'])]
            # adjust to the frequency (above is per second) --> FLOPS/SMCycle
            math_bw = math_bw / gpu_config['sm_max_freq'] * query['avg_freq'] # TFLOP/s at target frequency
            dram_bw = gpu_config['dram_bw'] # --> DRAM clock is fixed
            estimated_time = query['mem_footprint'] / (dram_bw * 10 ** 6) if query['bound'] == 'mem' else query['flop'] / (math_bw * 10 ** 9) # to ms
        elif (op_type[0] == 'layernorm') or (op_type[0] == 'softmax') or (op_type[0] == 'elementwise'):
            dram_bw = gpu_config['dram_bw'] # --> DRAM clock is fixed
            estimated_time = query['mem_footprint'] / (dram_bw * 10 ** 6)

        estimated_energy = estimated_time * gpu_config['power_cap'] / 1000.

        return (estimated_time, estimated_energy)
                
