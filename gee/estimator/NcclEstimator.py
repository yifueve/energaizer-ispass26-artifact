import os
import numpy as np
import pandas as pd
import math
import copy
import csv
import json
import re
import yaml

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('..')

from gee.optimization_utils import optimize

from gee.estimator.BaseEstimator import BaseEstimator

class NcclEstimator(BaseEstimator):
    def __init__(self, lut_config=None, gpu_config=None, \
                 dvfs_aware=False, dvfs_inference_mode='single_source', dvfs_supply_voltage={}, dvfs_idle_power={}, \
                 lut_folder_abs_path='/mnt/c/Users/KyungmiLee/Documents/gpu-energy-estimation/lut'):
        
        op = 'nccl'
        ops_supported = ['nccl']
        super().__init__(op, ops_supported, gpu_config=gpu_config, \
                         dvfs_aware=dvfs_aware, dvfs_inference_mode=dvfs_inference_mode, dvfs_supply_voltage=dvfs_supply_voltage, dvfs_idle_power=dvfs_idle_power)

        self.model_database = {}

        for key in lut_config.keys():
            if key not in ops_supported:
                continue
            for key, value in lut_config['nccl'].items():
                self.model_database[('nccl', key)] = pd.read_csv(os.path.join(lut_folder_abs_path, value))

    def build(self):
        pass

    def predict(self, query, op, target_freq, verbose=False, root0_only=False, **kwargs):
        # query: {'op': 'all_gather', 'size': xx bytes, 'n_gpus': 2}
        # For NCCL, we will linearly interpolate from neighboring points in the lut

        lut = self.model_database[('nccl', query['n_gpus'])]
        
        # 1. Get all entries in lut corresponding to this operation
        if root0_only and 'root' in lut.columns:
            df_op = lut.loc[(lut['op'] == query['op']) & (lut['root'] == 0) & (lut['inplace'] == query['inplace'])] 
        elif 'inplace' in lut.columns:
            df_op = lut.loc[(lut['op'] == query['op']) & (lut['inplace'] == query['inplace'])]
        else:
            df_op = lut.loc[(lut['op'] == query['op'])]

        # 2. Get all reference points (in size)
        df_op['log10_size'] = df_op['size'].apply(lambda x : math.log10(x))

        ref_sizes = df_op['log10_size'].unique()
        ref_sizes = np.sort(ref_sizes)

        # 3. Get neighboring entries 
        if ref_sizes.shape[0] == 0:
            raise ValueError("ERROR: This lookup table doesn't have any entry corresponding to the operation {}.".format(query['op']))

        neighboring_size_log10 = []
        neighboring_energy_per_bit = [0, 0]
        for idx in range(ref_sizes.shape[0]):
            # print(idx, math.log10(query['size']), ref_sizes[idx])
            # Left edge: size smaller than the smallest entry in the lut
            if (idx == 0) and (math.log10(query['size']) < ref_sizes[idx]):
                neighboring_size_log10 = [ref_sizes[idx], ref_sizes[idx + 1]]
                break
            
            # right edge: size larger than the largest entry in the lut
            if (idx == ref_sizes.shape[0] - 1) and (math.log10(query['size']) >= ref_sizes[idx]):
                neighboring_size_log10 = [ref_sizes[idx - 1], ref_sizes[idx]]
                break

            # between
            if (math.log10(query['size']) >= ref_sizes[idx]) and (math.log10(query['size']) < ref_sizes[idx + 1]):
                neighboring_size_log10 = [ref_sizes[idx], ref_sizes[idx + 1]]
                break
        
        energy_left = df_op.loc[df_op['log10_size'] == neighboring_size_log10[0]]
        energy_right = df_op.loc[df_op['log10_size'] == neighboring_size_log10[1]]

        for i in range(query['n_gpus']):
            neighboring_energy_per_bit[0] += energy_left['energy_per_bit_{}'.format(i)].mean()
            neighboring_energy_per_bit[1] += energy_right['energy_per_bit_{}'.format(i)].mean()

        time_left = energy_left['time'].values[0]
        time_right = energy_right['time'].values[0]
        neighboring_time = [time_left, time_right]
        
        # 4. Interpolate linearly (log-log for numerical stability)
        # x -> log10 of size, y -> log10 of energy
        neighboring_energy_per_bit = [math.log10(x) for x in neighboring_energy_per_bit]
        # print(neighboring_energy_per_bit)
        if verbose:
            print('neighboring sizes (log10): {}'.format(neighboring_size_log10))
            print('neighboring energy per bit: {}'.format(neighboring_energy_per_bit))
        slope = (neighboring_energy_per_bit[1] - neighboring_energy_per_bit[0]) / (neighboring_size_log10[1] - neighboring_size_log10[0])
        intercept = (neighboring_size_log10[1] * neighboring_energy_per_bit[0] - neighboring_size_log10[0] * neighboring_energy_per_bit[1]) / (neighboring_size_log10[1] - neighboring_size_log10[0])

        estimated_energy = 10 ** (slope * math.log10(query['size']) + intercept) # energy per bit

        neighboring_time = [math.log10(x) for x in neighboring_time]
        slope = (neighboring_time[1] - neighboring_time[0]) / (neighboring_size_log10[1] - neighboring_size_log10[0])
        intercept = (neighboring_size_log10[1] * neighboring_time[0] - neighboring_size_log10[0] * neighboring_time[1]) / (neighboring_size_log10[1] - neighboring_size_log10[0])

        estimated_time = 10 ** (slope * math.log10(query['size']) + intercept) # energy per bit

        return (estimated_time, -1 , estimated_energy * query['size'] * 8)
