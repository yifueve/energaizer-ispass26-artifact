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

class BaseAnalyticalModel():
    def __init__(self, op, ops_supported, gpu_config, \
                 dvfs_supply_voltage=None, dvfs_idle_power=None, \
                 dvfs_max_hbm_freq=2619, dvfs_max_core_freq=1980, dvfs_max_core_voltage=1400, dvfs_max_hbm_bw=3500, \
                 extrapolation_sm_base=108):
        self.op = op
        self.ops_supported = ops_supported
        self.gpu_config = gpu_config
        self.dvfs_supply_voltage = dvfs_supply_voltage
        self.dvfs_idle_power = dvfs_idle_power
        self.dvfs_max_hbm_freq = dvfs_max_hbm_freq
        self.dvfs_max_core_freq = dvfs_max_core_freq
        self.dvfs_max_core_voltage = dvfs_max_core_voltage
        self.dvfs_max_hbm_bw = dvfs_max_hbm_bw
        self.extrapolation_sm_base = extrapolation_sm_base

        self.dvfs = (dvfs_supply_voltage is not None) and (dvfs_idle_power is not None)

    def prepare(self, df):
        # df should be pandas dataframe

        # waves and tail effect
        df['max_concurrent_block'] = df['max_concurrent_block'].apply(lambda x: max(x, 1))
        df['n_waves'] = df.apply(lambda row: math.ceil(row['total_block_tiles'] / (self.gpu_config['num_sm'] * row['max_concurrent_block'])), axis=1)
        
        df['n_busy_sm'] = (df['total_block_tiles'] - 1) % self.gpu_config['num_sm'] + 1
        df['n_lazy_sm'] = self.gpu_config['num_sm'] - df['n_busy_sm']

        df['n_blocks_busy'] = df['total_block_tiles'].apply(lambda x: math.ceil(x / self.gpu_config['num_sm']))
        df['n_blocks_lazy'] = df['total_block_tiles'].apply(lambda x: math.floor(x / self.gpu_config['num_sm']))

        df['last_wave_blocks_busy'] = df.apply(lambda row: (row['n_blocks_busy'] - 1) % row['max_concurrent_block'] + 1, axis=1)
        df['last_wave_blocks_lazy'] = df.apply(lambda row: row['n_blocks_lazy'] % row['max_concurrent_block'], axis=1)

        # SM and SMSP level warp information
        df['sm_warps_full_capacity'] = df['n_warps_per_block'] * df['max_concurrent_block']
        df['sm_warps_last_wave_busy'] = df['n_warps_per_block'] * df['last_wave_blocks_busy']
        df['sm_warps_last_wave_lazy'] = df['n_warps_per_block'] * df['last_wave_blocks_lazy']

        df['smsp_n_warps_busy_full_capacity'] = df['sm_warps_full_capacity'].apply(lambda x: math.ceil(x/4)) # math.ceil(df['sm_warps_full_capacity'] / 4)
        df['smsp_n_warps_lazy_full_capacity'] =  df['sm_warps_full_capacity'].apply(lambda x: math.floor(x/4))
        df['smsp_n_warps_busy_last_wave_busy'] = df['sm_warps_last_wave_busy'].apply(lambda x: math.ceil(x/4))
        df['smsp_n_warps_lazy_last_wave_busy'] = df['sm_warps_last_wave_busy'].apply(lambda x: math.floor(x/4))
        df['smsp_n_warps_busy_last_wave_lazy'] = df['sm_warps_last_wave_lazy'].apply(lambda x: math.ceil(x/4))
        df['smsp_n_warps_lazy_last_wave_lazy'] = df['sm_warps_last_wave_lazy'].apply(lambda x: math.floor(x/4))

        df['n_smsp_busy_full_capacity'] = (df['sm_warps_full_capacity'] - 1) % 4 + 1
        df['n_smsp_lazy_full_capacity'] = 4 - df['n_smsp_busy_full_capacity']
        df['n_smsp_busy_last_wave_busy'] = (df['sm_warps_last_wave_busy'] - 1) % 4 + 1
        df['n_smsp_lazy_last_wave_busy'] = 4 - df['n_smsp_busy_last_wave_busy']
        df['n_smsp_busy_last_wave_lazy'] = (df['sm_warps_last_wave_lazy'] - 1) % 4 + 1
        df['n_smsp_lazy_last_wave_lazy'] = 4 - df['n_smsp_busy_last_wave_lazy']

        if self.dvfs:
            # Precompute dvfs related info
            # df['hbm_freq_norm'] = self.gpu_config['dram_freq'] / self.dvfs_max_hbm_freq # normalize by freq
            df['hbm_freq_norm'] = self.gpu_config['dram_bw'] / self.dvfs_max_hbm_bw # noralize by bw

            df['gpu_supply_voltage'] = df['avg_freq'].apply(lambda x: self.dvfs_supply_voltage[str(int((x - 210) / 15) * 15 + 210)]) # mV
            max_voltage = self.dvfs_max_core_voltage # max(dvfs_supply_voltage.values())
            df['gpu_supply_voltage_norm'] = df['gpu_supply_voltage'].apply(lambda x: x / max_voltage)
            max_freq = self.dvfs_max_core_freq # max([float(x) for x in dvfs_supply_voltage.keys()])
            df['gpu_freq_norm'] = df['avg_freq'].apply(lambda x: x / max_freq)
            df['dvfs_scale_factor'] = df['gpu_supply_voltage_norm'] ** 2 * df['gpu_freq_norm'] # V^2 * f
            df['idle_power'] = df['avg_freq'].apply(lambda x: self.dvfs_idle_power[str(int((x - 210) / 15) * 15 + 210)]) # W
            
            # SM Scaling Factors
            df['extrapolation_sm_scale_factor'] = self.gpu_config['num_sm'] / self.extrapolation_sm_base

            # TC scaling factor
            flops_per_tc = self.gpu_config['tc_bf16_flops'] / (self.gpu_config['num_sm'] * 4 * self.gpu_config['sm_max_freq']) * 10**6
            df['extrapolation_tc_scale_factor'] = flops_per_tc / 512
        
        df['gpu_sm'] = self.gpu_config['num_sm']


    def perf_model(self):
        raise NotImplementedError()
    
    def power_model(self):
        raise NotImplementedError()
    
    def model(self):
        raise NotImplementedError()

