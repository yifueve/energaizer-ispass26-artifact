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
                'dram': 466.3, \
                'fp_op': 10, \
                'xu_op': 100}

"""
Unlike GEMM (or Gemm-like such as COnv2d), nonlinear kernels don't have sophisticated latency hiding like buffering, etc.
Also, ILP can be limited due to data-dependency between subsequent operations.
They have synchronization frequently to guarantee data dependency.
Thus, we opt for a simple modeling, where we record the granularity of action (data, computation), ilp, and iteration counts.
"""
class NonlinearAnalyticalModel(BaseAnalyticalModel):
    def __init__(self, gpu_config, \
                 dvfs_supply_voltage=None, dvfs_idle_power=None, \
                 dvfs_max_hbm_freq=2619, dvfs_max_core_freq=1980, dvfs_max_core_voltage=1400):
        op = 'nonlinear'
        ops_supported = ['softmax', 'layernorm']
        super().__init__(op, ops_supported, gpu_config, \
                         dvfs_supply_voltage=dvfs_supply_voltage, dvfs_idle_power=dvfs_idle_power, \
                         dvfs_max_hbm_freq=dvfs_max_hbm_freq, dvfs_max_core_freq=dvfs_max_core_freq, dvfs_max_core_voltage=dvfs_max_core_voltage)

        self.stage_names = {}
        self.stage_names['softmax'] = ['max', 'exp', 'epilogue']
        self.stage_names['layernorm'] = ['stats', 'reduction', 'epilogue']

    def softmax_action_count(self, df, adjust_gmem_to_sm_ratio=1, adjust_sm_to_gmem_ratio=1):
        df['kernel_type_cunn'] = df['kernel_type'].apply(lambda x: x in ['cunn_SoftMaxForward', 'cunn_SoftMaxForwardSmem'])
        df['bytes_per_prec'] = df['prec'].apply(lambda x: prec_to_precision_bits(x) // 8)

        # df['softmax_dim_per_iter'] = df['warp_tile_softmax'] / df['num_warp_tile_softmax_temporal']

        # Memory: Gmem, Smem
        # Computation: Fp (ordinary floating-point operations), Xu (special operations like exponentials and division)

        # Max
        df['softmax_dim_per_iter'] = df['warp_tile_softmax'] / df['num_warp_tile_softmax_temporal']
        df['max_gmem_to_reg_granularity_bytes'] = df.apply(lambda row: (row['warp_tile_batch'] * row['softmax_dim_per_iter'] * row['bytes_per_prec']) if row['kernel_type_cunn'] \
                                                                    else (row['warp_tile_batch'] * row['warp_tile_softmax'] * row['bytes_per_prec']), axis=1)
        df['max_smem_to_reg_granularity_bytes'] = 0
        df['max_reg_to_gmem_granularity_bytes'] = 0
        df['max_reg_to_smem_granularity_bytes'] = df.apply(lambda row: row['max_gmem_to_reg_granularity_bytes'] if row['kernel_type'] == 'cunn_SoftMaxForwardSmem' else 0, axis=1)

        # Adjust for fused cases
        df['max_gmem_to_reg_granularity_bytes'] *= adjust_gmem_to_sm_ratio
        df['max_reg_to_smem_granularity_bytes'] *= adjust_gmem_to_sm_ratio

        df['max_fp_inst_per_element'] = 1 # max (comparision)
        df['max_xu_inst_per_element'] = 0

        df['max_fp_inst_effective_ilp_per_thread_per_batch'] = 1 # max operation has dependency within each thread, cannot be simply parallelized
        df['max_fp_inst_effective_ilp_per_thread'] = df['max_fp_inst_effective_ilp_per_thread_per_batch'] * df['warp_tile_batch']
        df['max_xu_inst_effective_ilp_per_thread_per_batch'] = 0
        df['max_xu_inst_effective_ilp_per_thread'] = 0

        df['max_memory_iter'] = df.apply(lambda row: row['num_warp_tile_softmax_temporal'] if row['kernel_type_cunn'] \
                                                    else 1, axis=1)
        df['max_fp_inst_iter'] = df.apply(lambda row: row['ilp'] * row['num_warp_tile_softmax_temporal'] if row['kernel_type_cunn'] \
                                                    else row['num_warp_tile_softmax_temporal'], axis=1)
        df['max_xu_inst_iter'] = 0
        
        df['max_reduction_smem_access_per_warp'] = df['kernel_type_cunn'].apply(lambda x: 2 if x else 0)

        # Exp
        df['exp_gmem_to_reg_granularity_bytes'] = df.apply(lambda row: row['max_gmem_to_reg_granularity_bytes'] if row['kernel_type'] == 'cunn_SoftMaxForward' else 0, axis=1)
        df['exp_smem_to_reg_granularity_bytes'] = df.apply(lambda row: row['max_gmem_to_reg_granularity_bytes'] if row['kernel_type'] == 'cunn_SoftMaxForwardSmem' else 0, axis=1)
        df['exp_reg_to_smem_granularity_bytes'] = 0
        df['exp_reg_to_gmem_granularity_bytes'] = 0

        # Adjust for fused cases
        df['exp_gmem_to_reg_granularity_bytes'] *= adjust_gmem_to_sm_ratio
        df['exp_smem_to_reg_granularity_bytes'] *= adjust_gmem_to_sm_ratio

        df['exp_fp_inst_per_element'] = 2 # subtract, accumulate
        df['exp_xu_inst_per_element'] = 1 # exponential

        df['exp_fp_inst_effective_ilp_per_thread_per_batch'] = df.apply(lambda row: (row['ilp'], 1) if row['kernel_type_cunn'] \
                                                                                    else (row['num_warp_tile_softmax_temporal'], 1), axis=1)
        df['exp_fp_inst_effective_ilp_per_thread'] = df.apply(lambda row: (row['exp_fp_inst_effective_ilp_per_thread_per_batch'][0] * row['warp_tile_batch'], \
                                                                        row['exp_fp_inst_effective_ilp_per_thread_per_batch'][1] * row['warp_tile_batch']), axis=1)
        df['exp_xu_inst_effective_ilp_per_thread_per_batch'] = df.apply(lambda row: row['ilp'] if row['kernel_type_cunn'] \
                                                                                    else row['num_warp_tile_softmax_temporal'], axis=1)
        df['exp_xu_inst_effective_ilp_per_thread'] = df['exp_xu_inst_effective_ilp_per_thread_per_batch'] * df['warp_tile_batch']

        df['exp_memory_iter'] = df.apply(lambda row: row['num_warp_tile_softmax_temporal'] if row['kernel_type_cunn']\
                                                    else 1, axis=1)
        df['exp_fp_inst_iter'] = df.apply(lambda row: (row['num_warp_tile_softmax_temporal'], row['num_warp_tile_softmax_temporal'] * row['ilp']) if row['kernel_type_cunn'] \
                                                    else (1, row['num_warp_tile_softmax_temporal']), axis=1)
        df['exp_xu_inst_iter'] = df.apply(lambda row: row['num_warp_tile_softmax_temporal'] if row['kernel_type_cunn'] \
                                                    else 1, axis=1)
        
        df['exp_reduction_smem_access_per_warp'] = df['max_reduction_smem_access_per_warp']

        # Epilogue
        df['epilogue_gmem_to_reg_granularity_bytes'] = df.apply(lambda row: row['max_gmem_to_reg_granularity_bytes'] if row['kernel_type'] == 'cunn_SoftMaxForward' else 0, axis=1)
        df['epilogue_smem_to_reg_granularity_bytes'] = df.apply(lambda row: row['max_gmem_to_reg_granularity_bytes'] if row['kernel_type'] == 'cunn_SoftMaxForwardSmem' else 0, axis=1)
        df['epilogue_reg_to_smem_granularity_bytes'] = 0
        df['epilogue_reg_to_gmem_granularity_bytes'] = df['max_gmem_to_reg_granularity_bytes']

        # Adjust for fused cases
        df['epilogue_gmem_to_reg_granularity_bytes'] *= adjust_gmem_to_sm_ratio
        df['epilogue_smem_to_reg_granularity_bytes'] *= adjust_gmem_to_sm_ratio
        df['epilogue_reg_to_gmem_granularity_bytes'] *= adjust_sm_to_gmem_ratio

        df['epilogue_fp_inst_per_element'] = df['kernel_type_cunn'].apply(lambda x: 1 if x else 0)
        df['epilogue_xu_inst_per_element'] = df['kernel_type_cunn'].apply(lambda x: 2 if x else 1)

        df['epilogue_fp_inst_effective_ilp_per_thread_per_batch'] = df.apply(lambda row: row['ilp'] if row['kernel_type_cunn'] else row['num_warp_tile_softmax_temporal'], axis=1)
        df['epilogue_fp_inst_effective_ilp_per_thread'] = df['epilogue_fp_inst_effective_ilp_per_thread_per_batch'] * df['warp_tile_batch']
        df['epilogue_xu_inst_effective_ilp_per_thread_per_batch'] = df.apply(lambda row: row['ilp'] if row['kernel_type_cunn'] else row['num_warp_tile_softmax_temporal'], axis=1)
        df['epilogue_xu_inst_effective_ilp_per_thread'] = df['epilogue_xu_inst_effective_ilp_per_thread_per_batch'] * df['warp_tile_batch']

        df['epilogue_memory_iter'] = df.apply(lambda row: row['num_warp_tile_softmax_temporal'] if row['kernel_type_cunn']\
                                                    else 1, axis=1)
        df['epilogue_fp_inst_iter'] = df.apply(lambda row: row['num_warp_tile_softmax_temporal'] if row['kernel_type_cunn'] else 1, axis=1)
        df['epilogue_xu_inst_iter'] = df.apply(lambda row: row['num_warp_tile_softmax_temporal'] if row['kernel_type_cunn'] else 1, axis=1)

        df['epilogue_reduction_smem_access_per_warp'] = 0

    def layernorm_action_count(self, df, adjust_gmem_to_sm_ratio=1, adjust_sm_to_gmem_ratio=1):
        df['bytes_per_prec'] = df['prec'].apply(lambda x: prec_to_precision_bits(x) // 8)

        df['layernorm_dim_per_iter'] = df['warp_tile_layernorm'] / df['num_warp_tile_layernorm_temporal']

        # 1) Calculating mean and variance of data
        df['stats_gmem_to_reg_granularity_bytes'] = df['warp_tile_batch'] * df['layernorm_dim_per_iter'] * df['bytes_per_prec'] * adjust_gmem_to_sm_ratio
        df['stats_smem_to_reg_granularity_bytes'] = 0
        df['stats_reg_to_gmem_granularity_bytes'] = 0
        df['stats_reg_to_smem_granularity_bytes'] = 0 # ignore warp/block reduction smem accesses for mean/var info

        df['stats_fp_inst_per_element'] = 7 # 3 add/subtract operations, 2 fma operations
        df['stats_xu_inst_per_element'] = 1 # one inverse operation for every element

        df['stats_fp_inst_effective_ilp_per_thread_per_batch'] = 1 # due to dependency, operations cannot be parallelized
        df['stats_fp_inst_effective_ilp_per_thread'] = df['stats_fp_inst_effective_ilp_per_thread_per_batch'] * df['warp_tile_batch']
        df['stats_xu_inst_effective_ilp_per_thread_per_batch'] = 1 # due to dependency, operations cannot be parallelized
        df['stats_xu_inst_effective_ilp_per_thread'] = df['stats_xu_inst_effective_ilp_per_thread_per_batch'] * df['warp_tile_batch']

        df['stats_memory_iter'] = df['num_warp_tile_layernorm_temporal']
        df['stats_fp_inst_iter'] = df['ilp'] * df['num_warp_tile_layernorm_temporal']
        df['stats_xu_inst_iter'] = df['ilp'] * df['num_warp_tile_layernorm_temporal']

        df['stats_reduction_smem_access_per_warp'] = 0 # ignore

        # 2) Reduction and get the mean, std --> For each pair of thread, first reduce within a warp (16 -> 8 -> 4 -> 2 -> 1); then, reduce across warps (2 -> 1)
        df['reduction_gmem_to_reg_granularity_bytes'] = 0
        df['reduction_smem_to_reg_granularity_bytes'] = 0 # ignore any smem <-> reg traffics
        df['reduction_reg_to_smem_granularity_bytes'] = 0 # ignore any smem <-> reg traffics
        df['reduction_reg_to_gmem_granularity_bytes'] = 0 

        df['reduction_fp_inst_per_element'] = 10 # 10 floating point operations
        df['reduction_xu_inst_per_element'] = 1 # 1 inverse operation

        df['reduction_fp_inst_effective_ilp_per_thread_per_batch'] = 1 # ignore potential instruction-level parallelism
        df['reduction_fp_inst_effective_ilp_per_thread'] = df['reduction_fp_inst_effective_ilp_per_thread_per_batch'] * df['warp_tile_batch']
        df['reduction_xu_inst_effective_ilp_per_thread_per_batch'] = 1
        df['reduction_xu_inst_effective_ilp_per_thread'] = df['reduction_xu_inst_effective_ilp_per_thread_per_batch'] * df['warp_tile_batch']

        df['reduction_memory_iter'] = 0
        df['reduction_fp_inst_iter'] = (32 - 1) + (4 - 1) # (16 + 8 + 4 + 2 + 1) + (2 + 1) (assuming 32 threads/warp and 4 warps/block)
        df['reduction_xu_inst_iter'] = (32 - 1) + (4 - 1)

        df['reduction_reduction_smem_access_per_warp'] = 0 # ignore

        # 3) Epilogue
        df['epilogue_gmem_to_reg_granularity_bytes'] = df['warp_tile_batch'] * df['layernorm_dim_per_iter'] * df['bytes_per_prec'] * adjust_gmem_to_sm_ratio
        df['epilogue_smem_to_reg_granularity_bytes'] = 0
        df['epilogue_reg_to_gmem_granularity_bytes'] = df['warp_tile_batch'] * df['layernorm_dim_per_iter'] * df['bytes_per_prec'] * adjust_sm_to_gmem_ratio
        df['epilogue_reg_to_smem_granularity_bytes'] = 0

        df['epilogue_fp_inst_per_element'] = 4 # assume both beta and gamma exist
        df['epilogue_xu_inst_per_element'] = 0 # assume inverse of square root of variance has already been calculated

        df['epilogue_fp_inst_effective_ilp_per_thread_per_batch'] = df['ilp'] # no dependency
        df['epilogue_fp_inst_effective_ilp_per_thread'] = df['epilogue_fp_inst_effective_ilp_per_thread_per_batch'] * df['warp_tile_batch']
        df['epilogue_xu_inst_effective_ilp_per_thread_per_batch'] = 0
        df['epilogue_xu_inst_effective_ilp_per_thread'] = df['epilogue_xu_inst_effective_ilp_per_thread_per_batch'] * df['warp_tile_batch']

        df['epilogue_memory_iter'] = df['num_warp_tile_layernorm_temporal']
        df['epilogue_fp_inst_iter'] = df['num_warp_tile_layernorm_temporal']
        df['epilogue_xu_inst_iter'] = 0

        df['epilogue_reduction_smem_access_per_warp'] = 0 # ignore


    def perf_model(self, df, op_nonlinear, train=True, adjust_gmem_to_sm_ratio=1, adjust_sm_to_gmem_ratio=1):
        self.prepare(df)

        # Action count
        if op_nonlinear == 'softmax':
            self.softmax_action_count(df, adjust_gmem_to_sm_ratio, adjust_sm_to_gmem_ratio)
        elif op_nonlinear == 'layernorm':
            self.layernorm_action_count(df, adjust_gmem_to_sm_ratio, adjust_sm_to_gmem_ratio)
        else:
            raise NotImplementedError()
        
        # Cycle estimate
        df['dram_bw'] = df['avg_freq'].apply(lambda x : math.ceil(self.gpu_config['dram_bw'] * 10 ** 9 / (x * 10 ** 6)))

        fp_bw_per_sm = int(self.gpu_config['cuda_fp32_flops'] * (10 ** 12) / (self.gpu_config['sm_max_freq'] * (10 ** 6)) / self.gpu_config['num_sm'])
        df['fp_bw_per_sm'] = df['prec'].apply(lambda x: fp_bw_per_sm if x == 'fp32' else fp_bw_per_sm * 2)
        
        xu_bw_per_sm = self.gpu_config['xu_bw_per_sm']

        for tag in ['full_capacity', 'last_wave_busy', 'last_wave_lazy']:
            for op in self.stage_names[op_nonlinear]:

                # Memory related
                df['{}_gmem_to_reg_all_sms_{}'.format(op, tag)] = df.apply(lambda row: self.gpu_config['num_sm'] * row['sm_warps_full_capacity'] * row['{}_gmem_to_reg_granularity_bytes'.format(op)] if tag == 'full_capacity' else \
                                                                                    (row['n_busy_sm'] * row['sm_warps_last_wave_busy'] + row['n_lazy_sm'] * row['sm_warps_last_wave_lazy']) * row['{}_gmem_to_reg_granularity_bytes'.format(op)], axis=1)
                df['t_from_l2_{}_{}'.format(op, tag)] = df['{}_gmem_to_reg_all_sms_{}'.format(op, tag)].apply(lambda x: max(x / self.gpu_config['l2_bw'], inst_latency['l2']) if x > 0 else 0)
                df['t_from_dram_{}_{}'.format(op, tag)] = df.apply(lambda row: max(row['{}_gmem_to_reg_all_sms_{}'.format(op, tag)] / row['dram_bw'], inst_latency['dram']) if row['{}_gmem_to_reg_all_sms_{}'.format(op, tag)] > 0 else 0, axis=1)
                df['t_from_gmem_{}_{}'.format(op, tag)] = df.apply(lambda row: max(row['t_from_l2_{}_{}'.format(op, tag)], row['t_from_dram_{}_{}'.format(op, tag)]), axis=1)

                df['t_from_l2_{}_{}'.format(op, tag)] = df['t_from_l2_{}_{}'.format(op, tag)] * df['{}_memory_iter'.format(op)]
                df['t_from_dram_{}_{}'.format(op, tag)] = df['t_from_dram_{}_{}'.format(op, tag)] * df['{}_memory_iter'.format(op)]
                df['t_from_gmem_{}_{}'.format(op, tag)] = df['t_from_gmem_{}_{}'.format(op, tag)] * df['{}_memory_iter'.format(op)]

                df['{}_smem_to_reg_all_warps_{}'.format(op, tag)] = df.apply(lambda row: row['sm_warps_{}'.format(tag)] * row['{}_smem_to_reg_granularity_bytes'.format(op)], axis=1)
                df['t_smem2reg_{}_{}'.format(op, tag)] = df['{}_smem_to_reg_all_warps_{}'.format(op, tag)].apply(lambda x: max(x / 128, inst_latency['shared']) if x > 0 else 0)
                df['t_smem2reg_{}_{}'.format(op, tag)] = df['t_smem2reg_{}_{}'.format(op, tag)] * df['{}_memory_iter'.format(op)]

                df['{}_reg_to_smem_all_warps_{}'.format(op, tag)] = df.apply(lambda row: row['sm_warps_{}'.format(tag)] * row['{}_reg_to_smem_granularity_bytes'.format(op)], axis=1)
                df['t_reg2smem_{}_{}'.format(op, tag)] = df['{}_reg_to_smem_all_warps_{}'.format(op, tag)].apply(lambda x: max(x / 128, inst_latency['shared']) if x > 0 else 0)
                df['t_reg2smem_{}_{}'.format(op, tag)] = df['t_reg2smem_{}_{}'.format(op, tag)] * df['{}_memory_iter'.format(op)]

                df['{}_reg_to_gmem_all_sms_{}'.format(op, tag)] = df.apply(lambda row: self.gpu_config['num_sm'] * row['sm_warps_full_capacity'] * row['{}_reg_to_gmem_granularity_bytes'.format(op)] if tag == 'full_capacity' else \
                                                                                    (row['n_busy_sm'] * row['sm_warps_last_wave_busy'] + row['n_lazy_sm'] * row['sm_warps_last_wave_lazy']) * row['{}_reg_to_gmem_granularity_bytes'.format(op)], axis=1)
                df['t_to_l2_{}_{}'.format(op, tag)] = df['{}_reg_to_gmem_all_sms_{}'.format(op, tag)].apply(lambda x: max(x / self.gpu_config['l2_bw'], inst_latency['l2']) if x > 0 else 0)
                df['t_to_l2_{}_{}'.format(op, tag)] = df['t_to_l2_{}_{}'.format(op, tag)] * df['{}_memory_iter'.format(op)]
                df['t_to_dram_{}_{}'.format(op, tag)] = df.apply(lambda row: max(row['{}_reg_to_gmem_all_sms_{}'.format(op, tag)] / row['dram_bw'], inst_latency['dram']) if row['{}_reg_to_gmem_all_sms_{}'.format(op, tag)] > 0 else 0, axis=1)
                df['t_to_dram_{}_{}'.format(op, tag)] = df['t_to_dram_{}_{}'.format(op, tag)] * df['{}_memory_iter'.format(op)]

                # FP core related
                def get_time(inst_per_element, ilp_per_thread, warps_per_sm, iter, bw, latency):
                    if type(ilp_per_thread) != tuple:
                        ilp_per_thread = tuple([ilp_per_thread] * inst_per_element)
                    if type(iter) != tuple:
                        iter = tuple([iter] * inst_per_element)
                    
                    t = 0
                    for i in range(inst_per_element):
                        n_concurrent_inst = warps_per_sm * 32 * ilp_per_thread[i]
                        _t = max(n_concurrent_inst / bw, latency)
                        t += _t * iter[i]
                    
                    return t
                
                df['t_fp_inst_{}_{}'.format(op, tag)] = df.apply(lambda row: get_time(row['{}_fp_inst_per_element'.format(op)], row['{}_fp_inst_effective_ilp_per_thread'.format(op)], \
                                                                                    row['sm_warps_{}'.format(tag)], row['{}_fp_inst_iter'.format(op)], row['fp_bw_per_sm'], inst_latency['fp_op']), axis=1)
            
                # XU core related
                df['t_xu_inst_{}_{}'.format(op, tag)] = df.apply(lambda row: get_time(row['{}_xu_inst_per_element'.format(op)], row['{}_xu_inst_effective_ilp_per_thread'.format(op)], \
                                                                                    row['sm_warps_{}'.format(tag)], row['{}_xu_inst_iter'.format(op)], xu_bw_per_sm, inst_latency['xu_op']), axis=1)
            

                # Reduction
                df['t_reduction_{}_{}'.format(op, tag)] = df.apply(lambda row: max(row['sm_warps_{}'.format(tag)] * row['{}_reduction_smem_access_per_warp'.format(op)] / 128, inst_latency['shared']) * 2 if row['{}_reduction_smem_access_per_warp'.format(op)] > 0 else 0, axis=1)

                # For one wave, group to all/gmem/smem/fp/xu
                df['t_estimated_one_wave_{}_{}'.format(op, tag)] = df['t_from_gmem_{}_{}'.format(op, tag)] + df['t_smem2reg_{}_{}'.format(op, tag)] + df['t_reg2smem_{}_{}'.format(op, tag)] + df['t_to_l2_{}_{}'.format(op, tag)] + \
                                                                   df['t_fp_inst_{}_{}'.format(op, tag)] + df['t_xu_inst_{}_{}'.format(op, tag)] + df['t_reduction_{}_{}'.format(op, tag)]
                df['t_estimated_one_wave_gmem_{}_{}'.format(op, tag)] = df['t_from_gmem_{}_{}'.format(op, tag)] + df['t_to_l2_{}_{}'.format(op, tag)]
                df['t_estimated_one_wave_smem_{}_{}'.format(op, tag)] = df['t_smem2reg_{}_{}'.format(op, tag)] + df['t_reg2smem_{}_{}'.format(op, tag)] + df['t_reduction_{}_{}'.format(op, tag)]
                df['t_estimated_one_wave_dram_{}_{}'.format(op, tag)] = df['t_from_dram_{}_{}'.format(op, tag)] + df['t_to_dram_{}_{}'.format(op, tag)]
                df['t_estimated_one_wave_l2_{}_{}'.format(op, tag)] = df['t_from_l2_{}_{}'.format(op, tag)] + df['t_to_l2_{}_{}'.format(op, tag)]

            # Calculate for one wave, sum across ops
            df['t_estimated_one_wave_gmem_{}'.format(tag)] = 0
            df['t_estimated_one_wave_smem_{}'.format(tag)] = 0
            df['t_estimated_one_wave_fp_inst_{}'.format(tag)] = 0
            df['t_estimated_one_wave_xu_inst_{}'.format(tag)] = 0
            df['t_estimated_one_wave_dram_{}'.format(tag)] = 0
            df['t_estimated_one_wave_l2_{}'.format(tag)] = 0
            for op in self.stage_names[op_nonlinear]:
                df['t_estimated_one_wave_gmem_{}'.format(tag)] += df['t_estimated_one_wave_gmem_{}_{}'.format(op, tag)]
                df['t_estimated_one_wave_smem_{}'.format(tag)] += df['t_estimated_one_wave_smem_{}_{}'.format(op, tag)]
                df['t_estimated_one_wave_fp_inst_{}'.format(tag)] += df['t_fp_inst_{}_{}'.format(op, tag)]
                df['t_estimated_one_wave_xu_inst_{}'.format(tag)] += df['t_xu_inst_{}_{}'.format(op, tag)]
                df['t_estimated_one_wave_dram_{}'.format(tag)] += df['t_estimated_one_wave_dram_{}_{}'.format(op, tag)]
                df['t_estimated_one_wave_l2_{}'.format(tag)] += df['t_estimated_one_wave_l2_{}_{}'.format(op, tag)]

        # Sum across ops and resources
        for op in self.stage_names[op_nonlinear]:
            df['t_estimated_{}'.format(op)] = (df['n_waves'] - 1) * df['t_estimated_one_wave_{}_full_capacity'.format(op)] + df['t_estimated_one_wave_{}_last_wave_busy'.format(op)]
        
        df['t_estimated_gmem'] = (df['n_waves'] - 1) * df['t_estimated_one_wave_gmem_full_capacity'] + df['t_estimated_one_wave_gmem_last_wave_busy']
        df['t_estimated_smem'] = (df['n_waves'] - 1) * df['t_estimated_one_wave_smem_full_capacity'] + df['t_estimated_one_wave_smem_last_wave_busy']
        df['t_estimated_fp_inst'] = (df['n_waves'] - 1) * df['t_estimated_one_wave_fp_inst_full_capacity'] + df['t_estimated_one_wave_fp_inst_last_wave_busy']
        df['t_estimated_xu_inst'] = (df['n_waves'] - 1) * df['t_estimated_one_wave_xu_inst_full_capacity'] + df['t_estimated_one_wave_xu_inst_last_wave_busy']
        df['t_estimated_dram'] = (df['n_waves'] - 1) * df['t_estimated_one_wave_dram_full_capacity'] + df['t_estimated_one_wave_dram_last_wave_busy']
        df['t_estimated_l2'] = (df['n_waves'] - 1) * df['t_estimated_one_wave_l2_full_capacity'] + df['t_estimated_one_wave_l2_last_wave_busy']

        df['t_estimated'] = 0
        for op in self.stage_names[op_nonlinear]:
            df['t_estimated'] += df['t_estimated_{}'.format(op)]

        if train:
            df['cycles'] = df.apply(lambda row: math.ceil(row['time'] * 10**3 * row['avg_freq']), axis=1)
            # df['cycles'] = df['elapsed_cycles']

    def power_model(self, df):

        # compute activity factors (w/o correction) here
        # a_dram (or a_l2) = t_estimated_dram (or l2) / t_estimated 
        # --> correction: final time = t_gmem * lambda_gmem + ...
        # -->             a_dram_corrected = t_estimated_dram * lambda_gmem / t_final 
        # -->                              = a_dram * (t_estimated / t_final) * lambda_gmem
        # a_fp (or xu) consider both busy and lazy
        # a_fp = (n_busy_sm * a_fp_busy + n_lazy_sm * a_fp_lazy) / total_sm
        # a_fp_busy = t_estimated_fp_inst / t_estimated
        # a_fp_lazy = (n_waves - 1) * t_estimated_one_wave_fp_inst_full_capacity + t_estimated_one_wave_fp_inst_last_wave_lazy / t_estimated
        # ...
        
        df['a_dram_precorrection'] = df['t_estimated_dram'] / df['t_estimated']
        df['a_l2_precorrection'] = df['t_estimated_l2'] / df['t_estimated']

        df['a_smem_busy'] = df['t_estimated_smem'] / df['t_estimated']
        df['a_smem_lazy'] = ((df['n_waves'] - 1) * df['t_estimated_one_wave_smem_full_capacity'] + df['t_estimated_one_wave_smem_last_wave_lazy']) / df['t_estimated']
        df['a_smem_precorrection'] = (df['a_smem_busy'] * df['n_busy_sm'] + df['a_smem_lazy'] * df['n_lazy_sm']) / self.gpu_config['num_sm']

        df['a_fp_busy'] = df['t_estimated_fp_inst'] / df['t_estimated']
        df['a_fp_lazy'] = ((df['n_waves'] - 1) * df['t_estimated_one_wave_fp_inst_full_capacity'] + df['t_estimated_one_wave_fp_inst_last_wave_lazy']) / df['t_estimated']
        df['a_fp_precorrection'] = (df['a_fp_busy'] * df['n_busy_sm'] + df['a_fp_lazy'] * df['n_lazy_sm']) / self.gpu_config['num_sm']
        
        df['a_xu_busy'] = df['t_estimated_xu_inst'] / df['t_estimated']
        df['a_xu_lazy'] = ((df['n_waves'] - 1) * df['t_estimated_one_wave_xu_inst_full_capacity'] + df['t_estimated_one_wave_xu_inst_last_wave_lazy']) / df['t_estimated']
        df['a_xu_precorrection'] = (df['a_xu_busy'] * df['n_busy_sm'] + df['a_xu_lazy'] * df['n_lazy_sm']) / self.gpu_config['num_sm']
        
    def model(self, df, op_nonlinear, train=True, adjust_gmem_to_sm_ratio=1, adjust_sm_to_gmem_ratio=1):
        self.perf_model(df, op_nonlinear, train, adjust_gmem_to_sm_ratio, adjust_sm_to_gmem_ratio)
        self.power_model(df)

