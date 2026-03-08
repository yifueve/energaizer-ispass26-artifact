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

mathinst_special_rules = {
    ('fp16', 'fp32', 16, 16, 16) : (16, 8, 16),
    ('fp16', 'fp32', 16, 16, 8) : (16, 8, 8),    # sliced-k 16, 16, 16 --> 16, 16, 8
    ('bf16', 'fp32', 16, 16, 16) : (16, 8, 16),
    ('bf16', 'fp32', 16, 16, 8) : (16, 8, 8),    # sliced-k 16, 16, 16 --> 16, 16, 8
    ('bf16', 'bf16', 16, 16, 16) : (16, 8, 16),
    ('bf16', 'bf16', 16, 16, 8) : (16, 8, 8)    # sliced-k 16, 16, 16 --> 16, 16, 8
}

inst_latency = {'shared': 29, \
                'mma.sync_16_8_8': 17.5, \
                'mma.sync_16_8_16': 26, \
                'fma': 20, \
                'l2': 261.5, \
                'dram': 466.3}

class GemmLikeAnalyticalModel(BaseAnalyticalModel):
    def __init__(self, gpu_config, \
                 dvfs_supply_voltage=None, dvfs_idle_power=None, \
                 dvfs_max_hbm_freq=2619, dvfs_max_core_freq=1980, dvfs_max_core_voltage=1400):
        op = 'gemm_like'
        ops_supported = ['gemm', 'conv2d', 'fmha-approximate']
        super().__init__(op, ops_supported, gpu_config, \
                         dvfs_supply_voltage=dvfs_supply_voltage, dvfs_idle_power=dvfs_idle_power, \
                         dvfs_max_hbm_freq=dvfs_max_hbm_freq, dvfs_max_core_freq=dvfs_max_core_freq, dvfs_max_core_voltage=dvfs_max_core_voltage)

    def prepare_for_gemm(self, df):
        # Number of 'multistages' and the last stage
        df['n_multistage'] = df.apply(lambda row: math.ceil(row['stagesK'] / row['multistageK']), axis=1)
        df['last_n_stages'] = df.apply(lambda row: (row['stagesK'] - 1) % row['multistageK'] + 1, axis=1)

        # Tile quantization effect
        # 1) CTA-level
        df['effective_block_tile_M'] = df.apply(lambda row: min(row['block_tile_M'], row['dimM']), axis=1)
        df['effective_block_tile_N'] = df.apply(lambda row: min(row['block_tile_N'], row['dimN']), axis=1)
        df['effective_block_tile_K'] = df.apply(lambda row: min(row['block_tile_K'], row['dimK']), axis=1)

        # 2) Stages
        df['last_stage_effective_block_tile_K'] = df.apply(lambda row: (row['dimK'] - 1) % row['block_tile_K'] + 1, axis=1)
        df['last_stage_groupsK'] = df.apply(lambda row: math.ceil(math.ceil(row['last_stage_effective_block_tile_K'] / row['num_warp_tile_K']) / row['warp_tile_K']), axis=1)

        # 3) SliceK special cases
        # effective_block_tile_K / num_warp_tile_K 
        df['effective_warp_tile_M'] = df.apply(lambda row: min(row['warp_tile_M'], row['block_tile_M']), axis=1)
        df['effective_warp_tile_N'] = df.apply(lambda row: min(row['warp_tile_N'], row['block_tile_N']), axis=1)
        df['effective_warp_tile_K'] = df.apply(lambda row: min(row['warp_tile_K'], row['block_tile_K']) if not row['sliceK'] else math.ceil(row['block_tile_K'] / row['num_warp_tile_K']), axis=1)

    def gemm_action_count(self, df, precM='bf16', precA='bf16', use_tensorcore=True, \
                          adjust_smem_to_gmem_ratio=1, adjust_gmem_to_smem_ratio=1, \
                          adjust_gmem_to_smem_select=None):
        
        # gmem, smem, flops action counting based on the parsed loopnest (cta, warp, inst)
        bytes_per_word_M = int(prec_to_precision_bits(precM) / 8)
        bytes_per_word_A = int(prec_to_precision_bits(precA) / 8)

        # 1. block -> per block tile size
        df['block_gmem_to_smem_per_block'] = (df['effective_block_tile_M'] + df['effective_block_tile_N']) * df['effective_block_tile_K'] * bytes_per_word_M
        df['block_gmem_to_smem_per_block_last_stage'] = (df['effective_block_tile_M'] + df['effective_block_tile_N']) * df['last_stage_effective_block_tile_K'] * bytes_per_word_M

        # 2. warptile -> per warptile tile size
        df['warptile_smem_to_reg_per_warptile'] = (df['effective_warp_tile_M'] + df['effective_warp_tile_N']) * df['effective_warp_tile_K'] * bytes_per_word_M

        # 3. mathinst -> instructions per warptile
        df['flops_per_warptile'] = df['warp_tile_M'] * df['warp_tile_N'] * df['warp_tile_K'] * 2 

        # 4. warptile -> reg to shared, per warptile
        df['warptile_reg_to_smem_per_warptile'] = (df['effective_warp_tile_M'] * df['effective_warp_tile_N']) * bytes_per_word_A

        # 5. block -> shared to global, per block
        df['block_smem_to_gmem_per_block'] = (df['effective_block_tile_M'] * df['effective_block_tile_N']) * bytes_per_word_A

        # 6. post-procesing for fused kernels
        if adjust_smem_to_gmem_ratio != 1:
            df['block_smem_to_gmem_per_block'] *= adjust_smem_to_gmem_ratio
        if adjust_gmem_to_smem_ratio != 1:
            if adjust_gmem_to_smem_select is None:
                df['block_gmem_to_smem_per_block'] *= adjust_gmem_to_smem_ratio
                df['block_gmem_to_smem_per_block_last_stage'] *= adjust_gmem_to_smem_ratio
            elif adjust_gmem_to_smem_select == 'A':
                df['block_gmem_to_smem_per_block'] = (adjust_gmem_to_smem_ratio * df['effective_block_tile_M'] + df['effective_block_tile_N']) * df['effective_block_tile_K'] * bytes_per_word_M
                df['block_gmem_to_smem_per_block_last_stage'] = (adjust_gmem_to_smem_ratio * df['effective_block_tile_M'] + df['effective_block_tile_N']) * df['last_stage_effective_block_tile_K'] * bytes_per_word_M
            elif adjust_gmem_to_smem_select == 'B':
                df['block_gmem_to_smem_per_block'] = (df['effective_block_tile_M'] + adjust_gmem_to_smem_ratio * df['effective_block_tile_N']) * df['effective_block_tile_K'] * bytes_per_word_M
                df['block_gmem_to_smem_per_block_last_stage'] = (df['effective_block_tile_M'] + adjust_gmem_to_smem_ratio * df['effective_block_tile_N']) * df['last_stage_effective_block_tile_K'] * bytes_per_word_M
            
        # 7. get throughput-based cycle estimates 
        # 7.1. block -> get total #blocks x #stagesK to determine gmem -> smem amount
        # -- L2 bandwidth, DRAM bandwidth evaluated separately
        # -- actual execution: DRAM -> L2 -> Shared
        df['block_gmem_to_smem'] = df['block_gmem_to_smem_per_block'] * df['total_block_tiles'] * (df['stagesK'] - 1) + df['block_gmem_to_smem_per_block_last_stage'] * df['total_block_tiles']
        df['cycle_block_gmem_to_smem_l2'] = df['block_gmem_to_smem'] / self.gpu_config['l2_bw']
        df['cycle_block_gmem_to_smem_dram'] = df.apply(lambda row: row['block_gmem_to_smem'] / (self.gpu_config['dram_bw'] * 10 ** 3 / row['avg_freq']), axis=1) # df['block_gmem_to_smem'] / (gpu_config['dram_bw'] * 10**9 / (df['avg_freq'] * 10**6))
        
        # 7.2. warptile -> get total n_blocks_busy x #stagesK x #warptiles x #groups to determine smem -> regfile amount (per SM, not per GPU)
        df['warptile_smem_to_reg'] = df['warptile_smem_to_reg_per_warptile'] * df['n_blocks_busy'] * df['num_warp_tile_M'] * df['num_warp_tile_N'] * df['num_warp_tile_K'] * ((df['stagesK'] - 1) * df['groupsK'] + df['last_stage_groupsK'])
        df['cycle_warptile_smem_to_reg'] = df['warptile_smem_to_reg'] / 128 # 128 bytes / cycle

        # 7.3. math instructions -> get total n_blocks_busy x #stagesK x #warptiles x #groups to determine flops per SM
        df['flops'] = df['flops_per_warptile'] * df['n_blocks_busy'] * df['num_warp_tile_M'] * df['num_warp_tile_N'] * df['num_warp_tile_K'] * ((df['stagesK'] - 1) * df['groupsK'] + df['last_stage_groupsK']) 
        
        gpu_max_bw_tc = self.gpu_config['{}_{}_flops'.format('tc' if use_tensorcore else 'cuda', precM)]
        ops_per_cycle = int(gpu_max_bw_tc * (10 ** 12) / (self.gpu_config['sm_max_freq'] * (10 ** 6)) / self.gpu_config['num_sm'])
        df['cycle_flops'] = df['flops'] / ops_per_cycle

        # 7.4. warptile -> get total n_blocks_busy x #warptiles (per SM)
        df['warptile_reg_to_smem'] = df['warptile_reg_to_smem_per_warptile'] * df['n_blocks_busy'] * df['num_warp_tile_M'] * df['num_warp_tile_N']
        df['cycle_warptile_reg_to_smem'] = df['warptile_reg_to_smem'] / 128 # 128 bytes / cycle

        # 7.5. block -> get total #blocks (per device)
        df['block_smem_to_gmem'] = df['block_smem_to_gmem_per_block'] * df['total_block_tiles']
        df['cycle_block_smem_to_gmem_l2'] = df['block_smem_to_gmem'] / self.gpu_config['l2_bw']
        df['cycle_block_smem_to_gmem_dram'] = df.apply(lambda row: row['block_smem_to_gmem'] / (self.gpu_config['dram_bw'] * 10**3 / row['avg_freq']), axis=1)

        # 7.6. get time by maxing throughputs
        df['cycle_smem'] = df['cycle_warptile_reg_to_smem'] + df['cycle_warptile_smem_to_reg']
        df['cycle_block_gmem_to_smem'] = df.apply(lambda row: max(row['cycle_block_gmem_to_smem_l2'], row['cycle_block_gmem_to_smem_dram']), axis=1)
        df['cycle_block_smem_to_gmem'] = df.apply(lambda row: max(row['cycle_block_smem_to_gmem_l2'], row['cycle_block_smem_to_gmem_dram']), axis=1)
        df['cycle_gmem'] = df['cycle_block_gmem_to_smem'] + df['cycle_block_smem_to_gmem']
        df['cycle_throughput_max'] = df.apply(lambda row: max(row['cycle_smem'], row['cycle_gmem'], row['cycle_flops']), axis=1)

    def model_sm_stage(self, df, precM='bf16', precA='bf16', use_tensorcore=True):
        # t_sm,stage: cycles required to process one 'stage' of block tile in SM
        # t_sm,stage = t_smem->reg,group + t_math,group + (groupsK - 1) * max(t_smem->reg,group, t_math,group)
        # t_smem->reg,group: cycles required to fetch one 'group' of warptile data (warp_tile_M * warp_tile_N * warp_tile_K) from shared to regfile
        # t_math,group: cycles required to complete math instructions on the 'group' of warptile data in regfile
        # we assume double-buffering at regfile level, hence max of these two. 

        tc_bw = self.gpu_config['{}_{}_flops'.format('tc' if use_tensorcore else 'cuda', precM)]
        tc_bw_per_smsp = int(tc_bw * (10 ** 12) / (self.gpu_config['sm_max_freq'] * (10 ** 6)) / self.gpu_config['num_sm'] / 4)
        
        df['mathinst_convert'] = df.apply(lambda row: (row['precM'], row['precA'], row['math_inst_M'], row['math_inst_N'], row['math_inst_K']) in mathinst_special_rules.keys(), axis=1)
        df['mathinst_converted'] = df.apply(lambda row: mathinst_special_rules[(row['precM'], row['precA'], row['math_inst_M'], row['math_inst_N'], row['math_inst_K'])] if row['mathinst_convert'] else (-1, -1, -1), axis=1)
        df['effective_math_inst_M'] = df.apply(lambda row: row['mathinst_converted'][0] if row['mathinst_convert'] else row['math_inst_M'], axis=1)
        df['effective_math_inst_N'] = df.apply(lambda row: row['mathinst_converted'][1] if row['mathinst_convert'] else row['math_inst_N'], axis=1)
        df['effective_math_inst_K'] = df.apply(lambda row: row['mathinst_converted'][2] if row['mathinst_convert'] else row['math_inst_K'], axis=1)
        df.drop(['mathinst_convert', 'mathinst_converted'], axis=1, inplace=True)


        # all of these analyses performed for full, last_busy, last_lazy
    
        for mode in ['full_capacity', 'last_wave_busy', 'last_wave_lazy']:

            # n_warps_per_sm = df['sm_warps_{}'.format(mode)]
            # n_warps_per_smsp_busy = df['smsp_n_warps_busy_{}'.format(mode)]

            # 1. t_smem->reg,group: assume that all warps active in a SM are concurrently loading from smem
            # -- then, throughput-based cycles is num_active_warps x smem_load_size / smem_bw (as smem is shared among all four smsps, not considering that here)
            # n_warps_per_sm * df['warptile_smem_to_reg_per_warptile'] / gpu_config['l2_bw']
            df['t_smem2reg_group_{}'.format(mode)] = df.apply(lambda row: max(row['sm_warps_{}'.format(mode)] * row['warptile_smem_to_reg_per_warptile'] / 128, inst_latency['shared']), axis=1)
            
            # last_wave_lazy no workload case
            if mode == 'last_wave_lazy':
                df['t_smem2reg_group_last_wave_lazy'] = df.apply(lambda row: row['t_smem2reg_group_last_wave_lazy'] if row['smsp_n_warps_busy_last_wave_lazy'] > 0 else 0, axis=1)

            #  2. t_math,group: this has to be considered per smsp, as tensorcore is not shared among different smsp
            # -- for one interation in k-dimension, 
            # -- # inst concurrently processed: (warptileM/mathinstM) * (warptileN/mathinstN) * n_warps_per_smsp_busy 
            # -- **change original mathinst to effectie inst considering special_rules (e.g., wmma 161616 --> mma.sync16816 x 2)
            # -- then, t_math,group = iterations across k dimension (warptileK/mathinstK) * max(warptileM * warptileN * mathinstK * n_warps_per_smsp_busy * 2 / tc_smsp_bw, latency)
            
            # TODO: Determine if 'effective_warp_tile' should be used for math instructions for both Tensor/CUDA Cores
            df['n_concurrent_math_inst_per_smsp_{}'.format(mode)] = df['smsp_n_warps_busy_{}'.format(mode)] * (df['effective_warp_tile_M'] / df['effective_math_inst_M']) * (df['effective_warp_tile_N'] / df['effective_math_inst_N'])
            df['t_math_group_{}'.format(mode)] = df.apply(lambda row: (row['warp_tile_K'] / row['effective_math_inst_K']) * max(row['n_concurrent_math_inst_per_smsp_{}'.format(mode)] * (2 * row['effective_math_inst_M'] * row['effective_math_inst_N'] * row['effective_math_inst_K']) / tc_bw_per_smsp, \
                                                                                                                                inst_latency['mma.sync_{}_{}_{}'.format(row['effective_math_inst_M'], row['effective_math_inst_N'], row['effective_math_inst_K'])] if (use_tensorcore and (not row['use_cuda_core_only'])) else inst_latency['fma']), axis=1)
            if mode == 'last_wave_lazy':
                df['t_math_group_last_wave_lazy'] = df.apply(lambda row: row['t_math_group_last_wave_lazy'] if row['smsp_n_warps_busy_last_wave_lazy'] > 0 else 0, axis=1)

            df['t_per_group_{}'.format(mode)] = df.apply(lambda row: max(row['t_smem2reg_group_{}'.format(mode)], row['t_math_group_{}'.format(mode)]), axis=1)
            
            df['t_sm_stage_{}'.format(mode)] = df['t_smem2reg_group_{}'.format(mode)] + df['t_math_group_{}'.format(mode)] + (df['groupsK'] - 1) * df['t_per_group_{}'.format(mode)]
            df['t_sm_last_stage_{}'.format(mode)] = df['t_smem2reg_group_{}'.format(mode)] + df['t_math_group_{}'.format(mode)] + (df['last_stage_groupsK'] - 1) * df['t_per_group_{}'.format(mode)]

    def model_mem_stagen(self, df, precM='bf16', precA='bf16', use_tensorcore=True, \
                         adjust_gmem_to_smem_ratio=1, adjust_gmem_to_smem_select=None):
        bytes_per_word_M = int(prec_to_precision_bits(precM) / 8)
        bytes_per_word_A = int(prec_to_precision_bits(precA) / 8)

        # t_mem_stage: cycles required to load N stages of block-level tiles from global to shared
        df['dram_bw'] = df['avg_freq'].apply(lambda x : math.ceil(self.gpu_config['dram_bw'] * 10 ** 9 / (x * 10 ** 6)))
        
        # analysis repeated for full_capacity or not / full_stage or not
        
        # 1. compute total amount (bytes) to load from global concurrently
        df['concurrent_global_load_bytes_full_capacity_full_stage'] = self.gpu_config['num_sm'] * (df['multistageK'] * df['max_concurrent_block']) * df['block_gmem_to_smem_per_block']
        df['concurrent_global_load_bytes_full_capacity_last_stage'] = self.gpu_config['num_sm'] * df['max_concurrent_block'] * (df['block_gmem_to_smem_per_block'] * (df['last_n_stages'] - 1) + df['block_gmem_to_smem_per_block_last_stage'])
        df['concurrent_global_load_bytes_last_wave_full_stage'] = (df['n_busy_sm'] * df['last_wave_blocks_busy'] + df['n_lazy_sm'] * df['last_wave_blocks_lazy']) * df['multistageK'] * df['block_gmem_to_smem_per_block']
        df['concurrent_global_load_bytes_last_wave_last_stage'] = (df['n_busy_sm'] * df['last_wave_blocks_busy'] + df['n_lazy_sm'] * df['last_wave_blocks_lazy']) * (df['block_gmem_to_smem_per_block'] * (df['last_n_stages'] - 1) + df['block_gmem_to_smem_per_block_last_stage'])

        # 2. TODO: determine the amount that will be l2 cache hit
        # https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/
        # threadblock swizzle -> best case scenario? same block tile is only fetched once from dram, then always hits in l2

        if adjust_gmem_to_smem_select is None:
            df['minimum_dram_traffic'] = df['batch'] * (df['dimM'] + df['dimN']) * df['dimK'] * bytes_per_word_M * adjust_gmem_to_smem_ratio
        elif adjust_gmem_to_smem_select == 'A':
            df['minimum_dram_traffic'] = df['batch'] * (df['dimM'] * adjust_gmem_to_smem_ratio + df['dimN']) * df['dimK'] * bytes_per_word_M 
        elif adjust_gmem_to_smem_select == 'B':
            df['minimum_dram_traffic'] = df['batch'] * (df['dimM'] + df['dimN'] * adjust_gmem_to_smem_ratio) * df['dimK'] * bytes_per_word_M 
        else:
            df['minimum_dram_traffic'] = df['batch'] * (df['dimM'] + df['dimN']) * df['dimK'] * bytes_per_word_M
        df['best_l2_cache_hit_from_block_tiles'] = (df['block_gmem_to_smem'] - df['minimum_dram_traffic']) / df['block_gmem_to_smem']    
        df['best_l2_cache_hit_from_block_tiles'] = df['best_l2_cache_hit_from_block_tiles'].apply(lambda x: max(x, 0))
        df['l2_cache_hit_ratio'] = df['best_l2_cache_hit_from_block_tiles'] # .apply(lambda x: max(0.15, x))

        for mode in ['full_capacity_full_stage', 'full_capacity_last_stage', 'last_wave_full_stage', 'last_wave_last_stage']:
            # 3. l2 cache hit / miss -> compute cycles separately
            # df['concurrent_l2_bytes_{}'.format(mode)] = df['concurrent_global_load_bytes_{}'.format(mode)] * df['l2_cache_hit_ratio']
            df['concurrent_dram_bytes_{}'.format(mode)] = df['concurrent_global_load_bytes_{}'.format(mode)] * (1 - df['l2_cache_hit_ratio'])

            # every data has to be fetched from L2 in theory
            df['t_from_l2_{}'.format(mode)] = df.apply(lambda row: max(row['concurrent_global_load_bytes_{}'.format(mode)] / self.gpu_config['l2_bw'], inst_latency['l2']), axis=1)
            df['t_from_dram_{}'.format(mode)] = df.apply(lambda row: max(row['concurrent_dram_bytes_{}'.format(mode)] / row['dram_bw'], inst_latency['dram'] * row['avg_freq'] / self.gpu_config['sm_max_freq']), axis=1)
            
            # 4. maximum of l2 and dram
            df['t_mem_stage_{}'.format(mode)] = df.apply(lambda row: max(row['t_from_l2_{}'.format(mode)], row['t_from_dram_{}'.format(mode)]), axis=1)

    def model_epilogue(self, df, precM='bf16', precA='bf16', use_tensorcore=True):
        # for slice-k kernels, there should be additional epilogue (TODO: same for split-K kernels)
        # slice-k -> partial sums in different warps have to be reduced together
        # --> written to smem all psums (st), reload to register for reduction operations (ld), re-write to smem the final result
        # --> write to smem is already covered in t_reg2smem_{} below
        # --> (ignoring reduction math operation itself for now: TODO), ld + st has to be added
        # --> ld: same amount as t_reg2smem_{}, st: t_reg2smem_{} / (sliceK amount)

        # t_end: with alpha=1, beta=0, this is just time for writing result back
        df['epilogue_global_bytes_full_capacity'] = df['block_smem_to_gmem_per_block'] * df['max_concurrent_block'] * self.gpu_config['num_sm']
        df['epilogue_global_bytes_last_wave'] = df['block_smem_to_gmem_per_block'] * (df['n_busy_sm'] * df['last_wave_blocks_busy'] + df['n_lazy_sm'] * df['last_wave_blocks_lazy'])

        # DRAM traffic among these
        # 1) For the last wave, it will be most likely residing in L2 - but if the capacity exceeds L2 capacity, the overflow should head to dram
        # 2) For the previous (non-last) waves, it will be evicted to dram
        df['dram_epilogue_global_bytes_full_capacity'] = df['epilogue_global_bytes_full_capacity']
        df['dram_epilogue_global_bytes_last_wave'] = df['epilogue_global_bytes_last_wave'].apply(lambda x: max(0, x - self.gpu_config['l2_size'] * 2**20))

        for mode in ['full_capacity', 'last_wave']:
            # first, regfile to shared (warps)
            df['t_reg2smem_{}'.format(mode)] = df.apply(lambda row: max(row['sm_warps_{}'.format('full_capacity' if mode=='full_capacity' else 'last_wave_busy')] * row['warptile_reg_to_smem_per_warptile'] / 128, inst_latency['shared']), axis=1)
            df['t_slicek_ld_{}'.format(mode)] = df.apply(lambda row: row['t_reg2smem_{}'.format(mode)] if row['sliceK'] else 0, axis=1)
            df['t_slicek_st_{}'.format(mode)] = df.apply(lambda row: max(row['sm_warps_{}'.format('full_capacity' if mode=='full_capacity' else 'last_wave_busy')] * row['warptile_reg_to_smem_per_warptile'] / (128 * row['num_warp_tile_K']), inst_latency['shared']) if row['sliceK'] else 0, axis=1)

            # last wave lazy
            if mode == 'last_wave':
                df['t_reg2smem_{}_lazy'.format(mode)] = df.apply(lambda row: max(row['sm_warps_{}'.format('last_wave_lazy')] * row['warptile_reg_to_smem_per_warptile'] / 128, inst_latency['shared']), axis=1)
                df['t_slicek_ld_{}_lazy'.format(mode)] = df.apply(lambda row: row['t_reg2smem_{}'.format(mode)] if row['sliceK'] else 0, axis=1)
                df['t_slicek_st_{}_lazy'.format(mode)] = df.apply(lambda row: max(row['sm_warps_{}'.format('last_wave_lazy')] * row['warptile_reg_to_smem_per_warptile'] / (128 * row['num_warp_tile_K']), inst_latency['shared']) if row['sliceK'] else 0, axis=1)

            # second, shared to l2 (blocks)
            df['t_to_l2_{}'.format(mode)] = df.apply(lambda row: max(row['epilogue_global_bytes_{}'.format(mode)] / self.gpu_config['l2_bw'], inst_latency['l2']), axis=1)        

            # (third?), data in l2 will be eventually evicted to dram (simply add?)
            df['t_to_dram_{}'.format(mode)] = df.apply(lambda row: max(row['dram_epilogue_global_bytes_{}'.format(mode)] / row['dram_bw'], inst_latency['dram'] * row['avg_freq'] / self.gpu_config['sm_max_freq']), axis=1)
            df['t_to_global_{}'.format(mode)] = df.apply(lambda row: max(row['t_to_l2_{}'.format(mode)], row['t_to_dram_{}'.format(mode)]), axis=1)
            df['t_end_per_wave_{}'.format(mode)] = df['t_reg2smem_{}'.format(mode)] + df['t_to_global_{}'.format(mode)] + df['t_slicek_ld_{}'.format(mode)] + df['t_slicek_st_{}'.format(mode)]

    def perf_model(self, df, precM='bf16', precA='bf16', use_tensorcore=True, \
                   adjust_smem_to_gmem_ratio=1, adjust_gmem_to_smem_ratio=1, adjust_gmem_to_smem_select=None, \
                   train=True):
        
        # Prepare dataframe for modeling
        self.prepare(df)
        self.prepare_for_gemm(df)

        # Action counting
        self.gemm_action_count(df, precM, precA, use_tensorcore, \
                               adjust_smem_to_gmem_ratio, adjust_gmem_to_smem_ratio, adjust_gmem_to_smem_select)
        
        # Model SM stage, gmem stage, epilogue stage
        self.model_sm_stage(df, precM, precA, use_tensorcore)
        self.model_mem_stagen(df, precM, precA, use_tensorcore, adjust_gmem_to_smem_ratio=adjust_gmem_to_smem_ratio, adjust_gmem_to_smem_select=adjust_gmem_to_smem_select)
        self.model_epilogue(df, precM, precA, use_tensorcore)

        # Final processing
        # per-wave, t_work
        df['t_work_per_wave_full_capacity'] = df.apply(lambda row: max(row['t_sm_stage_full_capacity'] * (row['stagesK'] - 1) + row['t_sm_last_stage_full_capacity'], \
                                                                    row['t_sm_stage_full_capacity']+ row['t_sm_last_stage_full_capacity'] + (row['t_mem_stage_full_capacity_full_stage'] * (row['n_multistage'] - 1) + row['t_mem_stage_full_capacity_last_stage'])) if row['stagesK'] > 1 else  row['t_sm_last_stage_full_capacity'], axis=1)
        df['t_work_per_wave_last_wave'] = df.apply(lambda row: max(row['t_sm_stage_last_wave_busy'] * (row['stagesK']-1) + row['t_sm_last_stage_last_wave_busy'], \
                                                                row['t_sm_stage_last_wave_busy'] + row['t_sm_last_stage_last_wave_busy'] + (row['t_mem_stage_last_wave_full_stage'] * (row['n_multistage'] - 1) + row['t_mem_stage_last_wave_last_stage'])) if row['stagesK'] > 1 else row['t_sm_last_stage_last_wave_busy'], axis=1)
        
        # per-wave, t_start
        # treat same as t_mem_stage
        df['t_start_per_wave_full_capacity'] = df.apply(lambda row: row['t_mem_stage_full_capacity_full_stage'] if row['stagesK'] >= row['multistageK'] else row['t_mem_stage_full_capacity_last_stage'], axis=1)
        df['t_start_per_wave_last_wave'] =  df.apply(lambda row: row['t_mem_stage_last_wave_full_stage'] if row['stagesK'] >= row['multistageK'] else row['t_mem_stage_last_wave_last_stage'], axis=1)

        # t_end
        df['t_wave_full_capacity'] = df['t_work_per_wave_full_capacity'] + df['t_start_per_wave_full_capacity'] + df['t_end_per_wave_full_capacity']
        df['t_wave_last_wave'] = df['t_work_per_wave_last_wave'] + df['t_start_per_wave_last_wave'] + df['t_end_per_wave_last_wave']

        # final
        df['t_timeline_estimated'] = df['t_wave_full_capacity'] * (df['n_waves'] - 1) + df['t_wave_last_wave']

        df['t_start'] = df['t_start_per_wave_full_capacity'] * (df['n_waves'] - 1) + df['t_start_per_wave_last_wave']
        df['t_work'] = df['t_work_per_wave_full_capacity'] * (df['n_waves'] - 1) + df['t_work_per_wave_last_wave']
        df['t_end'] = df['t_end_per_wave_full_capacity'] * (df['n_waves'] - 1) + df['t_end_per_wave_last_wave']

        df['block_waves'] = df['total_block_tiles'].apply(lambda x : math.ceil(x / self.gpu_config['num_sm']))
        df['many_K'] = df.apply(lambda row: row['totalK'] >= (row['multistageK'] * row['block_tile_K']), axis=1)
        df['many_blocks'] = df.apply(lambda row: row['n_blocks_busy'] > row['max_concurrent_block'], axis=1)
        if train:
            df['cycles'] = df.apply(lambda row: math.ceil(row['time'] * 10**3 * row['avg_freq']), axis=1) # Using the measured time divided by iterations
            # df['cycles'] = df['Elapsed Cycles'].apply(lambda x: eval(x.replace(',', ''))) # Using the NCU measurement

    def power_model(self, df):
        # activity factors to track: a_dram, a_l2, a_smem, a_math (either tensorcore or cuda fp/int units), a_other (constant term)
        # for lazy and busy sm, we have to track both -> weighted average later 

        # 1) t_start
        df['a_dram_start_full_capacity'] = df.apply(lambda row: (row['t_from_dram_full_capacity_full_stage'] if row['stagesK'] >= row['multistageK'] else row['t_from_dram_full_capacity_last_stage']) / row['t_start_per_wave_full_capacity'], axis=1)
        df['a_dram_start_last_wave'] = df.apply(lambda row: (row['t_from_dram_last_wave_full_stage'] if row['stagesK'] >= row['multistageK'] else row['t_from_dram_last_wave_last_stage']) / row['t_start_per_wave_last_wave'], axis=1)

        df['a_l2_start_full_capacity'] = df.apply(lambda row: (row['t_from_l2_full_capacity_full_stage'] if row['stagesK'] >= row['multistageK'] else row['t_from_l2_full_capacity_last_stage']) / row['t_start_per_wave_full_capacity'], axis=1)
        df['a_l2_start_last_wave'] = df.apply(lambda row: (row['t_from_l2_last_wave_full_stage'] if row['stagesK'] >= row['multistageK'] else row['t_from_l2_last_wave_last_stage']) / row['t_start_per_wave_last_wave'], axis=1)
        
        df['a_smem_start_full_capacity'] = 0
        df['a_smem_start_last_wave'] = 0
        df['a_math_start_full_capacity'] = 0
        df['a_math_start_last_wave'] = 0

        # 2) t_work
        df['a_dram_work_full_capacity'] = df.apply(lambda row: (row['t_from_dram_full_capacity_full_stage'] * (row['n_multistage'] - 1) + row['t_from_dram_full_capacity_last_stage']) / row['t_work_per_wave_full_capacity'], axis=1)
        df['a_dram_work_last_wave'] = df.apply(lambda row: (row['t_from_dram_last_wave_full_stage'] * (row['n_multistage'] - 1) + row['t_from_dram_last_wave_last_stage']) / row['t_work_per_wave_last_wave'], axis=1)

        df['a_l2_work_full_capacity'] = df.apply(lambda row: (row['t_from_l2_full_capacity_full_stage'] * (row['n_multistage'] - 1) + row['t_from_l2_full_capacity_last_stage']) / row['t_work_per_wave_full_capacity'], axis=1)
        df['a_l2_work_last_wave'] = df.apply(lambda row: (row['t_from_l2_last_wave_full_stage'] * (row['n_multistage'] - 1) + row['t_from_l2_last_wave_last_stage']) / row['t_work_per_wave_last_wave'], axis=1)
        
        df['a_smem_work_full_capacity'] = df.apply(lambda row: \
                                                (row['t_smem2reg_group_full_capacity'] * (row['groupsK'] * (row['stagesK'] - 1) + row['last_stage_groupsK'])) / row['t_work_per_wave_full_capacity'], axis=1)
        df['a_smem_work_last_wave_busy'] = df.apply(lambda row: \
                                                    (row['t_smem2reg_group_last_wave_busy'] * (row['groupsK'] * (row['stagesK'] - 1) + row['last_stage_groupsK'])) / row['t_work_per_wave_last_wave'], axis=1)
        df['a_smem_work_last_wave_lazy'] = df.apply(lambda row: \
                                                    (row['t_smem2reg_group_last_wave_lazy'] * (row['groupsK'] * (row['stagesK'] - 1) + row['last_stage_groupsK'])) / row['t_work_per_wave_last_wave'], axis=1)
        df['a_smem_work_last_wave'] = (df['n_busy_sm'] * df['a_smem_work_last_wave_busy'] + df['n_lazy_sm'] * df['a_smem_work_last_wave_lazy']) / (df['n_busy_sm'] + df['n_lazy_sm'])

        df['a_math_work_full_capacity'] = df.apply(lambda row: \
                                                (row['t_math_group_full_capacity'] * (row['groupsK'] * (row['stagesK'] - 1) + row['last_stage_groupsK'])) / row['t_work_per_wave_full_capacity'], axis=1)
        df['a_math_work_last_wave_busy'] = df.apply(lambda row: \
                                                (row['t_math_group_last_wave_busy'] * (row['groupsK'] * (row['stagesK'] - 1) + row['last_stage_groupsK'])) / row['t_work_per_wave_last_wave'], axis=1)
        df['a_math_work_last_wave_lazy'] = df.apply(lambda row: \
                                                (row['t_math_group_last_wave_lazy'] * (row['groupsK'] * (row['stagesK'] - 1) + row['last_stage_groupsK'])) / row['t_work_per_wave_last_wave'], axis=1)
        df['a_math_work_last_wave'] = (df['n_busy_sm'] * df['a_math_work_last_wave_busy'] + df['n_lazy_sm'] * df['a_math_work_last_wave_lazy']) / (df['n_busy_sm'] + df['n_lazy_sm'])

        # 3) t_end
        df['a_dram_end_full_capacity'] = df.apply(lambda row: (row['t_to_dram_full_capacity'] / row['t_end_per_wave_full_capacity']), axis=1)
        df['a_dram_end_last_wave'] = df.apply(lambda row: (row['t_to_dram_last_wave'] / row['t_end_per_wave_last_wave']), axis=1)

        df['a_l2_end_full_capacity'] = df.apply(lambda row: (row['t_to_l2_full_capacity'] / row['t_end_per_wave_full_capacity']), axis=1)
        df['a_l2_end_last_wave'] = df.apply(lambda row: (row['t_to_l2_last_wave'] / row['t_end_per_wave_last_wave']), axis=1)

        df['a_smem_end_full_capacity'] = df.apply(lambda row: (row['t_reg2smem_full_capacity'] + row['t_slicek_ld_full_capacity'] + row['t_slicek_st_full_capacity']) / row['t_end_per_wave_full_capacity'], axis=1)
        df['a_smem_end_last_wave_busy'] = df.apply(lambda row: (row['t_reg2smem_last_wave'] + row['t_slicek_ld_last_wave'] + row['t_slicek_st_last_wave']) / row['t_end_per_wave_last_wave'], axis=1)
        df['a_smem_end_last_wave_lazy'] = df.apply(lambda row: (row['t_reg2smem_last_wave_lazy'] + row['t_slicek_ld_last_wave_lazy'] + row['t_slicek_st_last_wave_lazy']) / row['t_end_per_wave_last_wave'], axis=1)
        df['a_smem_end_last_wave'] = (df['n_busy_sm'] * df['a_smem_end_last_wave_busy'] + df['n_lazy_sm'] * df['a_smem_end_last_wave_lazy']) / (df['n_busy_sm'] + df['n_lazy_sm'])

        df['a_math_end_full_capacity'] = 0
        df['a_math_end_last_wave'] = 0

    def model(self, df, \
              adjust_smem_to_gmem_ratio=1, adjust_gmem_to_smem_ratio=1, adjust_gmem_to_smem_select=None, \
              train=True):
        
        precM = df['precM'].values[0]
        precA = df['precA'].values[0]
        use_tensorcore = df['useTensorCore'].values[0]
        
        self.perf_model(df, precM, precA, use_tensorcore, \
                        adjust_smem_to_gmem_ratio, adjust_gmem_to_smem_ratio, adjust_gmem_to_smem_select, train)
        self.power_model(df)


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
    

# if __name__ == '__main__':
#     with open('/mnt/c/Users/KyungmiLee/Documents/gpu-energy-estimation/config/gpu/yz8.yaml', 'r') as f:
#         gpu_config = yaml.safe_load(f)['gpu_configs']
#     t = GemmLikeAnalyticalModel(gpu_config)

#     query = {}
#     query['kernel_name'] = 'void cutlass::Kernel<cutlass_80_tensorop_bf16_s16816gemm_bf16_64x256_32x4_nn_align8>(T1::Params)'
#     query['batch'] = 128
#     query['dimM'] = 1024
#     query['dimN'] = 1024
#     query['dimK'] = 128
#     query['grid_size'] = (32, 2, 128)
#     query['block_size'] = (128, 1, 1)
#     query['useTensorCore'] = True
#     query['max_concurrent_block'] = 2
#     query['avg_freq'] = 900
#     query['precM'] = 'bf16'
#     query['precA'] = 'bf16'
#     query['time'] = 0.5168630886556216

#     query['use_cuda_core_only'] = False
#     query['gemv'] = False
#     query['block_tile_M'] = 64
#     query['block_tile_N'] = 256
#     query['block_tile_K'] = 32
#     query['num_block_tile_batch'] = 128
#     query['num_block_tile_M'] = 16
#     query['num_block_tile_N'] = 4
#     query['num_block_tile_K'] = 4
#     query['total_block_tiles'] = 8192
#     query['splitK'] = -1
#     query['totalK'] = 128
#     query['splitK_batch'] = -1
#     query['stagesK'] = 4
#     query['multistageK'] = 4
#     query['threads'] = 128
#     query['n_warps_per_block'] = 4
#     query['warp_tile_M'] = 64
#     query['warp_tile_N'] = 64
#     query['warp_tile_K'] = 16
#     query['num_warp_tile_M'] = 1
#     query['num_warp_tile_N'] = 4
#     query['num_warp_tile_K'] = 1
#     query['math_inst_M'] = 16
#     query['math_inst_N'] = 8
#     query['math_inst_K'] = 16
#     query['sliceK'] = False
#     query['groupsK'] = 2

#     query = pd.DataFrame([query], index=[0])
#     t.model(query)
#     query = query.iloc[0].to_dict()

#     for key, value in query.items():
#         print('{}: {}'.format(key, value))

