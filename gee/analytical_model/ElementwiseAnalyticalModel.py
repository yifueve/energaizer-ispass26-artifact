import os

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
# import matplotlib
# import matplotlib.pyplot as plt

import re
import math
import yaml

import sys
sys.path.append('..')

from gee.analytical_model.BaseAnalyticalModel import BaseAnalyticalModel
from gee.optimization_utils import prec_to_precision_bits

inst_latency = {'shared': 29, \
                'mma.sync_16_8_8': 17.5, \
                'mma.sync_16_8_16': 26, \
                'fma': 20, \
                'l2': 261.5, \
                'dram': 466.3}

class ElementwiseAnalyticalModel(BaseAnalyticalModel):
    def __init__(self, gpu_config, \
                 dvfs_supply_voltage=None, dvfs_idle_power=None, \
                 dvfs_max_hbm_freq=2619, dvfs_max_core_freq=1980, dvfs_max_core_voltage=1400):
        op = 'elementwise'
        ops_supported = ['pointwise_mul', 'pointwise_add', \
                         'scalar_mul', 'scalar_add', \
                         'typecast_to_fp32', 'typecast_to_bf16', \
                         'relu', 'gelu', 'silu', 'tanh', 'sigmoid', \
                         'unspecified_activation', 'unspecified_tensor', 'unspecified_scalar']
        
        super().__init__(op, ops_supported, gpu_config, \
                         dvfs_supply_voltage=dvfs_supply_voltage, dvfs_idle_power=dvfs_idle_power, \
                         dvfs_max_hbm_freq=dvfs_max_hbm_freq, dvfs_max_core_freq=dvfs_max_core_freq, dvfs_max_core_voltage=dvfs_max_core_voltage)

    def action_count(self, df, adjust_gmem_to_sm_ratio=1, adjust_sm_to_gmem_ratio=1):
        df['bytes_per_precIn'] = df['prec'].apply(lambda x: prec_to_precision_bits(x) // 8)

        def determine_out_prec(row):
            if row['op'] == 'typecast_to_fp32':
                return 4
            elif row['op'] == 'typecast_to_bf16':
                return 2
            else:
                return prec_to_precision_bits(row['prec']) // 8
            
        df['bytes_per_precOut'] = df.apply(lambda row: determine_out_prec(row), axis=1)

        def determine_num_input(row):
            if row['op'] in ['pointwise_mul', 'pointwise_add', 'unspecified_tensor']:
                return 2
            else:
                return 1
            
        df['num_input_arg'] = df.apply(lambda row: determine_num_input(row), axis=1)

        df['gmem_to_reg_block_bytes'] = df['block_tile'] * df['num_input_arg'] * df['bytes_per_precIn'] * adjust_gmem_to_sm_ratio
        df['reg_to_gmem_block_bytes'] = df['block_tile'] * df['bytes_per_precOut'] * adjust_sm_to_gmem_ratio

        df['fp_inst_per_element'] = 1 # assume to be 1 (mul, add, clamp, typecast, ..)
        df['fp_inst_ilp_per_thread'] = df['elements_per_thread']
        df['fp_inst_per_block'] = df['fp_inst_per_element'] * df['elements_per_thread'] * df['n_warps_per_block'] * 32

    def perf_model(self, df, train=True, adjust_gmem_to_sm_ratio=1, adjust_sm_to_gmem_ratio=1):
        self.prepare(df)
        self.action_count(df, adjust_gmem_to_sm_ratio, adjust_sm_to_gmem_ratio)

        # Cycle estimate
        df['dram_bw'] = df['avg_freq'].apply(lambda x : math.ceil(self.gpu_config['dram_bw'] * 10 ** 9 / (x * 10 ** 6)))

        fp_bw_per_sm = int(self.gpu_config['cuda_fp32_flops'] * (10 ** 12) / (self.gpu_config['sm_max_freq'] * (10 ** 6)) / self.gpu_config['num_sm'])
        
        def determine_effective_prec(row):
            if row['op'] in ['typecast_to_fp32', 'typecast_to_bf16']:
                return 'fp32'
            else:
                return row['prec']
            
        df['effective_prec'] = df.apply(lambda row: determine_effective_prec(row), axis=1)
        df['fp_bw_per_sm'] = df['effective_prec'].apply(lambda x: fp_bw_per_sm if x == 'fp32' else fp_bw_per_sm * 2)

        for tag in ['full_capacity', 'last_wave_busy', 'last_wave_lazy']:

            if tag == 'full_capacity':
                mult_factor = df['sm_warps_full_capacity'] * self.gpu_config['num_sm']
            else:
                mult_factor = (df['sm_warps_last_wave_busy'] * df['n_busy_sm'] + df['sm_warps_last_wave_lazy'] * df['n_lazy_sm'])

            df['{}_gmem_to_reg'.format(tag)] = df['gmem_to_reg_block_bytes'] * mult_factor
            df['t_from_l2_{}'.format(tag)] = df['{}_gmem_to_reg'.format(tag)].apply(lambda x: max(x / self.gpu_config['l2_bw'], inst_latency['l2']) if x > 0 else 0)
            df['t_from_dram_{}'.format(tag)] = df.apply(lambda row: max(row['{}_gmem_to_reg'.format(tag)] / row['dram_bw'], inst_latency['dram']) if row['{}_gmem_to_reg'.format(tag)] > 0 else 0, axis=1)
            df['t_from_gmem_{}'.format(tag)] = df.apply(lambda row: max(row['t_from_l2_{}'.format(tag)], row['t_from_dram_{}'.format(tag)]), axis=1)
        
            df['{}_reg_to_gmem'.format(tag)] = df['reg_to_gmem_block_bytes'] * mult_factor
            df['t_to_l2_{}'.format(tag)] = df['{}_reg_to_gmem'.format(tag)].apply(lambda x: max(x / self.gpu_config['l2_bw'], inst_latency['l2']) if x > 0 else 0)
            df['t_to_dram_{}'.format(tag)] = df.apply(lambda row: max(row['{}_reg_to_gmem'.format(tag)] / row['dram_bw'], inst_latency['dram']) if row['{}_reg_to_gmem'.format(tag)] > 0 else 0, axis=1)
            df['t_to_gmem_{}'.format(tag)] = df.apply(lambda row: max(row['t_to_l2_{}'.format(tag)], row['t_to_dram_{}'.format(tag)]), axis=1)
        
            df['{}_fp_inst_per_sm'.format(tag)] = df['fp_inst_per_block'] * df['sm_warps_{}'.format(tag)]
            df['t_fp_inst_{}'.format(tag)] = df['{}_fp_inst_per_sm'.format(tag)].apply(lambda x: max(x / fp_bw_per_sm, inst_latency['fma']))

            df['t_estimated_one_wave_{}'.format(tag)] = df['t_from_gmem_{}'.format(tag)] + \
                                                        df['t_fp_inst_{}'.format(tag)] + \
                                                        df['t_to_gmem_{}'.format(tag)]
            
            df['t_estimated_one_wave_dram_{}'.format(tag)] = df['t_from_dram_{}'.format(tag)] + df['t_to_dram_{}'.format(tag)]
            df['t_estimated_one_wave_l2_{}'.format(tag)] = df['t_from_l2_{}'.format(tag)] + df['t_to_l2_{}'.format(tag)]
            df['t_estimated_one_wave_gmem_{}'.format(tag)] = df['t_from_gmem_{}'.format(tag)] + df['t_to_gmem_{}'.format(tag)]
            df['t_estimated_one_wave_fp_{}'.format(tag)] = df['t_fp_inst_{}'.format(tag)]
            # df['t_estimated_one_wave_{}'.format(tag)] = df.apply(lambda row: max(row['t_estimated_one_wave_gmem_{}'.format(tag)], \
            #                                                                      row['t_estimated_one_wave_fp_{}'.format(tag)]) + row['t_estimated_one_wave_gmem_{}'.format(tag)], axis=1)

        df['t_estimated'] = (df['n_waves'] - 1) * df['t_estimated_one_wave_full_capacity'] + df['t_estimated_one_wave_last_wave_busy']
        df['t_estimated_dram'] = (df['n_waves'] - 1) * df['t_estimated_one_wave_dram_full_capacity'] + df['t_estimated_one_wave_dram_last_wave_busy']
        df['t_estimated_l2'] = (df['n_waves'] - 1) * df['t_estimated_one_wave_l2_full_capacity'] + df['t_estimated_one_wave_l2_last_wave_busy']
        df['t_estimated_gmem'] = (df['n_waves'] - 1) * df['t_estimated_one_wave_gmem_full_capacity'] + df['t_estimated_one_wave_gmem_last_wave_busy']
        df['t_estimated_fp'] = (df['n_waves'] - 1) * df['t_estimated_one_wave_fp_full_capacity'] + df['t_estimated_one_wave_fp_last_wave_busy']

        if train:
            df['cycles'] = df.apply(lambda row: math.ceil(row['time'] * 10**3 * row['avg_freq']), axis=1)

    def power_model(self, df):
        df['a_dram_precorrection'] = df['t_estimated_dram'] / df['t_estimated']
        df['a_l2_precorrection'] = df['t_estimated_l2'] / df['t_estimated']
        
        df['a_fp_busy'] = df['t_estimated_fp'] / df['t_estimated']
        df['a_fp_lazy'] = ((df['n_waves'] - 1) * df['t_estimated_one_wave_fp_full_capacity'] + df['t_estimated_one_wave_fp_last_wave_lazy']) / df['t_estimated']
        df['a_fp_precorrection'] = (df['a_fp_busy'] * df['n_busy_sm'] + df['a_fp_lazy'] * df['n_lazy_sm']) / self.gpu_config['num_sm']

    def model(self, df, train=True, adjust_gmem_to_sm_ratio=1, adjust_sm_to_gmem_ratio=1):
        self.perf_model(df, train, adjust_gmem_to_sm_ratio, adjust_sm_to_gmem_ratio)
        self.power_model(df)
