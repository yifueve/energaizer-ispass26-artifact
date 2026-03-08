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

class FlashAttentionAnalyticalModel(BaseAnalyticalModel):
    def __init__(self, gpu_config, \
                 dvfs_supply_voltage=None, dvfs_idle_power=None, \
                 dvfs_max_hbm_freq=2619, dvfs_max_core_freq=1980, dvfs_max_core_voltage=1400):
        op = 'flashattention'
        ops_supported = ['flashattention_v2']
        super().__init__(op, ops_supported, gpu_config, \
                         dvfs_supply_voltage=dvfs_supply_voltage, dvfs_idle_power=dvfs_idle_power, \
                         dvfs_max_hbm_freq=dvfs_max_hbm_freq, dvfs_max_core_freq=dvfs_max_core_freq, dvfs_max_core_voltage=dvfs_max_core_voltage)

    def flashattn_action_count(self, df, precM='bf16', precA='bf16', precSoft='fp32', use_tensorcore=True):
        bytes_per_word_M = int(prec_to_precision_bits(precM) / 8)
        bytes_per_word_A = int(prec_to_precision_bits(precA) / 8)
        bytes_per_word_Soft = int(prec_to_precision_bits(precSoft) / 8)

        # Q block GMEM -> SMEM
        df['q_block_gmem_to_smem_per_block'] = df['block_r'] * df['head_dim'] * bytes_per_word_M

        # K, V block GMEM -> SMEM
        df['kv_block_gmem_to_smem_per_block'] = df['block_c'] * df['head_dim'] * bytes_per_word_M

        # GEMM warp tile SMEM -> reg
        df['QK_gemm_warptile_smem_to_reg_per_warptile'] = (df['QK_gemm_warp_tile_M'] + df['QK_gemm_warp_tile_N']) * df['QK_gemm_warp_tile_K'] * bytes_per_word_M

        # GEMM warp tile flops
        df['QK_gemm_flops_per_warptile'] = df['QK_gemm_warp_tile_M'] * df['QK_gemm_warp_tile_N'] * df['QK_gemm_warp_tile_K'] * 2

        # GEMM warp tile SMEM -> reg
        df['SV_gemm_warptile_smem_to_reg_per_warptile'] = (df['SV_gemm_warp_tile_M'] + df['SV_gemm_warp_tile_N']) * df['SV_gemm_warp_tile_K'] * bytes_per_word_M

        # GEMM warp tile flops
        df['SV_gemm_flops_per_warptile'] = df['SV_gemm_warp_tile_M'] * df['SV_gemm_warp_tile_N'] * df['SV_gemm_warp_tile_K'] * 2

        # GMEM warp tile reg -> SMEM
        df['gemm_warptile_reg_to_smem_per_warptile'] = 0 # (df['QK_gemm_warp_tile_M'] * df['QK_gemm_warp_tile_N']) * bytes_per_word_Soft

        # Data movements for SMEM -> reg
        df['softmax_smem_reg_traffic'] = 0 # df['block_r'] * df['block_c'] * bytes_per_word_Soft

        # Row-max and subtract flops
        df['softmax_rowmax_flops'] = df['block_r'] * df['block_c'] * 2 # comparison + subtract after reduction for all elements

        # Exponential flops
        df['softmax_exp_flops'] = df['block_r'] * df['block_c']

        # Add and reduce flops
        df['softmax_add_flops'] = df['block_r'] *  df['block_c']

        # Elementwise mult
        df['softmax_mult_flops'] = df['block_r'] * df['block_c']

        # Result block SMEM -> GMEM
        df['result_block_smem_to_gmem_per_block'] = df['block_r'] * df['head_dim'] * bytes_per_word_A

    def model_intra_sm_gemm(self, df, precM='bf16', precA='bf16', use_tensorcore=True):
        tc_bw = self.gpu_config['{}_{}_flops'.format('tc' if use_tensorcore else 'cuda', precM)]
        tc_bw_per_smsp = int(tc_bw * (10 ** 12) / (self.gpu_config['sm_max_freq'] * (10 ** 6)) / self.gpu_config['num_sm'] / 4)
        
        for mode in ['full_capacity', 'last_wave_busy', 'last_wave_lazy']:

            # n_warps_per_sm = df['sm_warps_{}'.format(mode)]
            # n_warps_per_smsp_busy = df['smsp_n_warps_busy_{}'.format(mode)]

            # QK
            # 1. t_smem->reg,group: assume that all warps active in a SM are concurrently loading from smem
            # -- then, throughput-based cycles is num_active_warps x smem_load_size / smem_bw (as smem is shared among all four smsps, not considering that here)
            # n_warps_per_sm * df['warptile_smem_to_reg_per_warptile'] / gpu_config['l2_bw']
            df['QK_t_smem2reg_group_{}'.format(mode)] = df.apply(lambda row: max(row['sm_warps_{}'.format(mode)] * row['QK_gemm_warptile_smem_to_reg_per_warptile'] / 128, inst_latency['shared']), axis=1)

            #  2. t_math,group: this has to be considered per smsp, as tensorcore is not shared among different smsp
            # -- for one interation in k-dimension, 
            # -- # inst concurrently processed: (warptileM/mathinstM) * (warptileN/mathinstN) * n_warps_per_smsp_busy 
            # -- **change original mathinst to effectie inst considering special_rules (e.g., wmma 161616 --> mma.sync16816 x 2)
            # -- then, t_math,group = iterations across k dimension (warptileK/mathinstK) * max(warptileM * warptileN * mathinstK * n_warps_per_smsp_busy * 2 / tc_smsp_bw, latency)
            
            # TODO: Determine if 'effective_warp_tile' should be used for math instructions for both Tensor/CUDA Cores
            df['QK_n_concurrent_math_inst_per_smsp_{}'.format(mode)] = df['smsp_n_warps_busy_{}'.format(mode)] * (df['QK_gemm_warp_tile_M'] / df['gemm_math_inst_M']) * (df['QK_gemm_warp_tile_N'] / df['gemm_math_inst_N'])
            df['QK_t_math_group_{}'.format(mode)] = df.apply(lambda row: (row['QK_gemm_warp_tile_K'] / row['gemm_math_inst_K']) * max(row['QK_n_concurrent_math_inst_per_smsp_{}'.format(mode)] * (2 * row['gemm_math_inst_M'] * row['gemm_math_inst_N'] * row['gemm_math_inst_K']) / tc_bw_per_smsp, \
                                                                                                                                inst_latency['mma.sync_{}_{}_{}'.format(row['gemm_math_inst_M'], row['gemm_math_inst_N'], row['gemm_math_inst_K'])]), axis=1)
            
            df['QK_t_per_group_{}'.format(mode)] = df.apply(lambda row: max(row['QK_t_smem2reg_group_{}'.format(mode)], row['QK_t_math_group_{}'.format(mode)]), axis=1)
            
            df['QK_t_gemm_{}'.format(mode)] = df['QK_t_smem2reg_group_{}'.format(mode)] + df['QK_t_math_group_{}'.format(mode)] + df['QK_groupsK'] * df['QK_t_per_group_{}'.format(mode)]

            # SV
            df['SV_t_smem2reg_group_{}'.format(mode)] = df.apply(lambda row: max(row['sm_warps_{}'.format(mode)] * row['SV_gemm_warptile_smem_to_reg_per_warptile'] / 128, inst_latency['shared']), axis=1)
            df['SV_n_concurrent_math_inst_per_smsp_{}'.format(mode)] = df['smsp_n_warps_busy_{}'.format(mode)] * (df['SV_gemm_warp_tile_M'] / df['gemm_math_inst_M']) * (df['SV_gemm_warp_tile_N'] / df['gemm_math_inst_N'])
            df['SV_t_math_group_{}'.format(mode)] = df.apply(lambda row: (row['SV_gemm_warp_tile_K'] / row['gemm_math_inst_K']) * max(row['SV_n_concurrent_math_inst_per_smsp_{}'.format(mode)] * (2 * row['gemm_math_inst_M'] * row['gemm_math_inst_N'] * row['gemm_math_inst_K']) / tc_bw_per_smsp, \
                                                                                                                                inst_latency['mma.sync_{}_{}_{}'.format(row['gemm_math_inst_M'], row['gemm_math_inst_N'], row['gemm_math_inst_K'])]), axis=1)
            
            df['SV_t_per_group_{}'.format(mode)] = df.apply(lambda row: max(row['SV_t_smem2reg_group_{}'.format(mode)], row['SV_t_math_group_{}'.format(mode)]), axis=1)
            
            df['SV_t_gemm_{}'.format(mode)] = df['SV_t_smem2reg_group_{}'.format(mode)] + df['SV_t_math_group_{}'.format(mode)] + df['SV_groupsK'] * df['SV_t_per_group_{}'.format(mode)]


    def model_gmem_tile(self, df):
        # t_mem_stage: cycles required to load N stages of block-level tiles from global to shared
        df['dram_bw'] = df['avg_freq'].apply(lambda x : math.ceil(self.gpu_config['dram_bw'] * 10 ** 9 / (x * 10 ** 6)))
        
        # analysis repeated for full_capacity or not / full_stage or not
        
        # 1. compute total amount (bytes) to load from global concurrently
        df['concurrent_global_load_bytes_full_capacity'] = self.gpu_config['num_sm'] * (df['max_concurrent_block']) * df['kv_block_gmem_to_smem_per_block'] * 2
        df['concurrent_global_load_bytes_last_wave'] = (df['n_busy_sm'] * df['last_wave_blocks_busy'] + df['n_lazy_sm'] * df['last_wave_blocks_lazy']) * df['kv_block_gmem_to_smem_per_block'] * 2
        
        # 2. L2 cache --> ignore

        for mode in ['full_capacity', 'last_wave']:
            df['concurrent_dram_bytes_{}'.format(mode)] = df['concurrent_global_load_bytes_{}'.format(mode)] * (1 / df['num_block_r']) # Assuming the ideal cache hit for K and V tiles

            # every data has to be fetched from L2 in theory
            df['t_from_l2_{}'.format(mode)] = df.apply(lambda row: max(row['concurrent_global_load_bytes_{}'.format(mode)] / self.gpu_config['l2_bw'], inst_latency['l2']), axis=1)
            df['t_from_dram_{}'.format(mode)] = df.apply(lambda row: max(row['concurrent_dram_bytes_{}'.format(mode)] / row['dram_bw'], inst_latency['dram'] * row['avg_freq'] / self.gpu_config['sm_max_freq']), axis=1)
            
            # 4. maximum of l2 and dram
            df['t_gmem_tile_{}'.format(mode)] = df.apply(lambda row: max(row['t_from_l2_{}'.format(mode)], row['t_from_dram_{}'.format(mode)]), axis=1)

    def model_softmax(self, df):
        
        # CUDA Core bw
        tc_bw = self.gpu_config['{}_{}_flops'.format('cuda', 'fp32')]
        tc_bw_per_sm = int(tc_bw * (10 ** 12) / (self.gpu_config['sm_max_freq'] * (10 ** 6)) / self.gpu_config['num_sm'])

        # Simple bandwidth modeling
        df['mult_factor_full_capacity'] = df['max_concurrent_block']
        df['mult_factor_last_wave_busy'] = df['last_wave_blocks_busy']
        df['mult_factor_last_wave_lazy'] = df['last_wave_blocks_lazy']
        
        for mode in ['full_capacity', 'last_wave_busy', 'last_wave_lazy']:
    
            # Total SMEM -> Reg (Max, Subtract, Exp, Mult)
            df['concurrent_softmax_smem_to_reg_{}'.format(mode)] = df['softmax_smem_reg_traffic'] * df['mult_factor_{}'.format(mode)] * 4

            # Total Reg -> SMEM (Initial result, Subtract, Exp, Mult)
            df['concurrent_softmax_reg_to_smem_{}'.format(mode)] = df['softmax_smem_reg_traffic'] * df['mult_factor_{}'.format(mode)] * 4

            # Total FLOPs
            df['concurrent_softmax_flops_{}'.format(mode)] = (df['softmax_rowmax_flops'] + df['softmax_exp_flops'] + df['softmax_add_flops'] + df['softmax_mult_flops']) * df['mult_factor_{}'.format(mode)]

            df['t_softmax_smem_reg_{}'.format(mode)] = df.apply(lambda row: max((row['concurrent_softmax_smem_to_reg_{}'.format(mode)] + row['concurrent_softmax_reg_to_smem_{}'.format(mode)])/ 128, inst_latency['shared']), axis=1)
            df['t_softmax_compute_{}'.format(mode)] = df.apply(lambda row: max(row['concurrent_softmax_flops_{}'.format(mode)]  / tc_bw_per_sm, inst_latency['fp_op']), axis=1)

            df['t_softmax_{}'.format(mode)] = df.apply(lambda row: max(row['t_softmax_smem_reg_{}'.format(mode)], row['t_softmax_compute_{}'.format(mode)]), axis=1)

    def model_stage(self, df):
        # gmem_tile vs. (gemm x 2 + softmax)
        df['t_stage_full_capacity'] = df.apply(lambda row: max(row['t_gmem_tile_full_capacity'], row['QK_t_gemm_full_capacity'] + row['SV_t_gemm_full_capacity'] + row['t_softmax_full_capacity']), axis=1)
        df['t_stage_last_wave'] = df.apply(lambda row: max(row['t_gmem_tile_last_wave'], row['SV_t_gemm_last_wave_busy'] + row['SV_t_gemm_last_wave_busy'] + row['t_softmax_last_wave_busy']), axis=1)

        df['t_work'] = (df['t_stage_full_capacity'] * (df['n_waves'] - 1) + df['t_stage_last_wave']) * df['iteration_stages']

    def model_prologue(self, df):
        # df['dram_bw'] = df['avg_freq'].apply(lambda x : math.ceil(self.gpu_config['dram_bw'] * 10 ** 9 / (x * 10 ** 6)))
        
        df['prologue_concurrent_gmem_load_full_capacity'] = self.gpu_config['num_sm'] * (df['max_concurrent_block']) * (df['q_block_gmem_to_smem_per_block'] + df['kv_block_gmem_to_smem_per_block'])
        df['prologue_concurrent_gmem_load_last_wave'] = (df['n_busy_sm'] * df['last_wave_blocks_busy'] + df['n_lazy_sm'] * df['last_wave_blocks_lazy']) * (df['q_block_gmem_to_smem_per_block'] + df['kv_block_gmem_to_smem_per_block'])
        
        df['t_prologue_dram_full_capacity'] = df.apply(lambda row: max(row['prologue_concurrent_gmem_load_full_capacity'] / row['dram_bw'], inst_latency['dram'] * row['avg_freq'] / self.gpu_config['sm_max_freq']), axis=1)
        df['t_prologue_dram_last_wave'] = df.apply(lambda row: max(row['prologue_concurrent_gmem_load_last_wave'] / row['dram_bw'], inst_latency['dram'] * row['avg_freq'] / self.gpu_config['sm_max_freq']), axis=1)
        df['t_prologue_l2_full_capacity'] = df.apply(lambda row: max(row['prologue_concurrent_gmem_load_full_capacity'] / self.gpu_config['l2_bw'], inst_latency['l2']), axis=1)
        df['t_prologue_l2_last_wave'] = df.apply(lambda row: max(row['prologue_concurrent_gmem_load_last_wave'] / self.gpu_config['l2_bw'], inst_latency['l2']), axis=1)

        df['t_prologue_full_capacity'] = df.apply(lambda row: max(row['t_prologue_dram_full_capacity'], row['t_prologue_l2_full_capacity']), axis=1)
        df['t_prologue_last_wave'] = df.apply(lambda row: max(row['t_prologue_dram_last_wave'], row['t_prologue_l2_last_wave']), axis=1)

        df['t_start'] = df['t_prologue_full_capacity'] * (df['n_waves'] - 1) + df['t_prologue_last_wave']

    def model_epilogue(self, df):
        df['epilogue_concurrent_gmem_store_full_capacity']  = self.gpu_config['num_sm'] * df['max_concurrent_block'] * df['result_block_smem_to_gmem_per_block']
        df['epilogue_concurrent_gmem_store_last_wave_busy'] = df['n_busy_sm'] * df['last_wave_blocks_busy'] * df['result_block_smem_to_gmem_per_block']
        df['epilogue_concurrent_gmem_store_last_wave_lazy'] = df['n_lazy_sm'] * df['last_wave_blocks_lazy'] * df['result_block_smem_to_gmem_per_block']
        df['epilogue_concurrent_gmem_store_last_wave'] = df['epilogue_concurrent_gmem_store_last_wave_busy'] + df['epilogue_concurrent_gmem_store_last_wave_lazy']

        for mode in ['full_capacity', 'last_wave']:
            if mode == 'last_wave':
                df['t_epilogue_reg_to_smem_last_wave'.format(mode)] = df.apply(lambda row: max(row['last_wave_blocks_busy'] * row['result_block_smem_to_gmem_per_block'] / 128, inst_latency['shared']), axis=1)
                df['t_epilogue_reg_to_smem_last_wave_busy'.format(mode)] = df.apply(lambda row: max(row['last_wave_blocks_busy'] * row['result_block_smem_to_gmem_per_block'] / 128, inst_latency['shared']), axis=1)
                df['t_epilogue_reg_to_smem_last_wave_lazy'.format(mode)] = df.apply(lambda row: max(row['last_wave_blocks_lazy'] * row['result_block_smem_to_gmem_per_block'] / 128, inst_latency['shared']), axis=1)
            else:
                df['t_epilogue_reg_to_smem_{}'.format(mode)] = df.apply(lambda row: max(row['result_block_smem_to_gmem_per_block'] * row['max_concurrent_block'] / 128, inst_latency['shared']), axis=1)

            df['t_epilogue_smem_to_gmem_l2_{}'.format(mode)] = df.apply(lambda row: max(row['epilogue_concurrent_gmem_store_{}'.format(mode)] / self.gpu_config['l2_bw'], inst_latency['l2']), axis=1)
            df['t_epilogue_smem_to_gmem_dram_{}'.format(mode)] = df.apply(lambda row: max(row['epilogue_concurrent_gmem_store_{}'.format(mode)] / row['dram_bw'], inst_latency['dram'] * row['avg_freq'] / self.gpu_config['sm_max_freq']), axis=1)
            df['t_epilogue_smem_to_gmem_{}'.format(mode)] = df.apply(lambda row: max(row['t_epilogue_smem_to_gmem_l2_{}'.format(mode)], row['t_epilogue_smem_to_gmem_dram_{}'.format(mode)]), axis=1)
            df['t_epilogue_{}'.format(mode)] = df['t_epilogue_reg_to_smem_{}'.format(mode)] + df['t_epilogue_smem_to_gmem_{}'.format(mode)]

        df['t_end'] = df['t_epilogue_full_capacity'] * (df['n_waves'] - 1) + df['t_epilogue_last_wave']

    def perf_model(self, df, precM='bf16', precA='bf16', use_tensorcore=True, train=True):
        self.prepare(df)
        self.flashattn_action_count(df, precM=precM, precA=precA, use_tensorcore=True)

        self.model_intra_sm_gemm(df, precM, precA, use_tensorcore)
        self.model_gmem_tile(df)
        self.model_softmax(df)
        self.model_stage(df)
        self.model_prologue(df)
        self.model_epilogue(df)

        df['t_estimated'] = df['t_start'] + df['t_work'] + df['t_end']
        df['block_waves'] = df['total_block_tiles'].apply(lambda x : math.ceil(x / self.gpu_config['num_sm']))
        if train:
            df['cycles'] = df.apply(lambda row: math.ceil(row['time'] * 10**3 * row['avg_freq']), axis=1) # Using the measured time divided by iterations
        
    def power_model(self, df):
        # activity factors for: dram, l2, smem, tc, fp

        # 1) t_start -> dram, l2 (TODO: should also add smem (as it is being written)?)
        df['a_dram_start_full_capacity'] = df['t_prologue_dram_full_capacity'] / df['t_prologue_full_capacity']
        df['a_dram_start_last_wave'] = df['t_prologue_dram_last_wave'] / df['t_prologue_last_wave']

        df['a_l2_start_full_capacity'] = df['t_prologue_l2_full_capacity'] / df['t_prologue_full_capacity']
        df['a_l2_start_last_wave'] = df['t_prologue_l2_last_wave'] / df['t_prologue_last_wave']

        df['a_smem_start_full_capacity'] = 0
        df['a_smem_start_last_wave'] = 0
        df['a_tc_start_full_capacity'] = 0
        df['a_tc_start_last_wave'] = 0
        df['a_fp_start_full_capacity'] = 0
        df['a_fp_start_last_wave'] = 0

        # 2) t_work -> dram, l2, smem, tc, fp
        df['a_dram_work_full_capacity'] = df['t_from_dram_full_capacity'] / df['t_stage_full_capacity']
        df['a_dram_work_last_wave'] = df['t_from_dram_last_wave'] / df['t_stage_last_wave']

        df['a_l2_work_full_capacity'] = df['t_from_l2_full_capacity'] / df['t_stage_full_capacity']
        df['a_l2_work_last_wave'] = df['t_from_l2_last_wave'] / df['t_stage_last_wave']

        df['a_smem_work_full_capacity'] = (df['QK_t_smem2reg_group_full_capacity'] * df['QK_groupsK'] + df['SV_t_smem2reg_group_full_capacity'] * df['SV_groupsK']  + df['t_softmax_smem_reg_full_capacity']) / df['t_stage_full_capacity']
        df['a_smem_work_last_wave_busy'] = (df['QK_t_smem2reg_group_last_wave_busy'] * df['QK_groupsK'] + df['SV_t_smem2reg_group_last_wave_busy'] * df['SV_groupsK'] + df['t_softmax_smem_reg_last_wave_busy']) / df['t_stage_last_wave']
        df['a_smem_work_last_wave_lazy'] = (df['QK_t_smem2reg_group_last_wave_lazy'] * df['QK_groupsK'] + df['SV_t_smem2reg_group_last_wave_lazy'] * df['SV_groupsK'] + df['t_softmax_smem_reg_last_wave_lazy']) / df['t_stage_last_wave']
        df['a_smem_work_last_wave'] = (df['n_busy_sm'] * df['a_smem_work_last_wave_busy'] + df['n_lazy_sm'] * df['a_smem_work_last_wave_lazy']) / (df['n_busy_sm'] + df['n_lazy_sm'])

        df['a_tc_work_full_capacity'] = ((df['QK_t_math_group_full_capacity'] * df['QK_groupsK']) + (df['SV_t_math_group_full_capacity'] * df['SV_groupsK'])) / df['t_stage_full_capacity']
        df['a_tc_work_last_wave_busy'] = ((df['QK_t_math_group_last_wave_busy'] * df['QK_groupsK']) + (df['SV_t_math_group_last_wave_busy'] * df['SV_groupsK'])) / df['t_stage_last_wave']
        df['a_tc_work_last_wave_lazy'] = ((df['QK_t_math_group_last_wave_lazy'] * df['QK_groupsK']) + (df['SV_t_math_group_last_wave_lazy'] * df['SV_groupsK'])) / df['t_stage_last_wave']
        df['a_tc_work_last_wave'] = (df['n_busy_sm'] * df['a_tc_work_last_wave_busy'] + df['n_lazy_sm'] * df['a_tc_work_last_wave_lazy']) / (df['n_busy_sm'] + df['n_lazy_sm'])
        
        df['a_fp_work_full_capacity'] = (df['t_softmax_compute_full_capacity']) / df['t_stage_full_capacity']
        df['a_fp_work_last_wave_busy'] = (df['t_softmax_compute_last_wave_busy']) / df['t_stage_last_wave']
        df['a_fp_work_last_wave_lazy'] = df['t_softmax_compute_last_wave_lazy'] / df['t_stage_last_wave']
        df['a_fp_work_last_wave'] = (df['n_busy_sm'] * df['a_fp_work_last_wave_busy'] + df['n_lazy_sm'] * df['a_fp_work_last_wave_lazy']) / (df['n_busy_sm'] + df['n_lazy_sm'])

        # 3) t_end -> dram, l2, smem
        df['a_dram_end_full_capacity'] = df['t_epilogue_smem_to_gmem_dram_full_capacity'] / df['t_epilogue_full_capacity']
        df['a_dram_end_last_wave'] = df['t_epilogue_smem_to_gmem_dram_last_wave'] / df['t_epilogue_last_wave']

        df['a_l2_end_full_capacity'] = df['t_epilogue_smem_to_gmem_l2_full_capacity'] / df['t_epilogue_full_capacity']
        df['a_l2_end_last_wave'] = df['t_epilogue_smem_to_gmem_l2_last_wave'] / df['t_epilogue_last_wave']

        df['a_smem_end_full_capacity'] = df['t_epilogue_reg_to_smem_full_capacity'] / df['t_epilogue_full_capacity']
        df['a_smem_end_last_wave_busy'] = df['t_epilogue_reg_to_smem_last_wave_busy'] / df['t_epilogue_last_wave']
        df['a_smem_end_last_wave_lazy'] = df['t_epilogue_reg_to_smem_last_wave_lazy'] / df['t_epilogue_last_wave']
        df['a_smem_end_last_wave'] = (df['n_busy_sm'] * df['a_smem_end_last_wave_busy'] + df['n_lazy_sm'] * df['a_smem_end_last_wave_lazy']) / (df['n_busy_sm'] + df['n_lazy_sm'])

        df['a_tc_end_full_capacity'] = 0
        df['a_tc_end_last_wave'] = 0

        df['a_fp_end_full_capacity'] = 0
        df['a_fp_end_last_wave'] = 0

    def model(self, df, train=True):
        precM = df['precM'].values[0]
        precA = df['precA'].values[0]
        use_tensorcore = df['useTensorCore'].values[0]

        self.perf_model(df, precM, precA, use_tensorcore, train)
        self.power_model(df)

