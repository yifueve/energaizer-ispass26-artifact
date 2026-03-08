import os
import sys
import pandas as pd
import yaml
import json

import warnings
warnings.filterwarnings('ignore')

from collections import Counter

class Estimator():
    def __init__(self, \
                 gpu_config_yaml, \
                 lut_config_yaml, \
                 lut_folder_abs_path='/mnt/c/Users/KyungmiLee/Documents/gpu-energy-estimation/lut', \
                 **kwargs):
        
        pass

    def predict_single_kernel(self):
        raise NotImplementedError()
    
    def predict_model(self, query_list, target_freq, verbose=False, use_precomputed_coeffs=False, **kwargs):

        if ('target_gpu_yaml_path' in kwargs.keys()) and (kwargs['target_gpu_yaml_path'] is not None):
            with open(kwargs['target_gpu_yaml_path'], 'r') as f:
                target_gpu_config = yaml.safe_load(f)['gpu_configs']
        else:
            target_gpu_config = None

        if ('target_dvfs_supply_voltage_path' in kwargs.keys()) and (kwargs['target_dvfs_supply_voltage_path'] is not None):
            with open(kwargs['target_dvfs_supply_voltage_path'], 'r') as f:
                target_dvfs_supply_voltage = json.load(f)
            if kwargs['target_dvfs_supply_voltage_constant'] > 0:
                for key, value in target_dvfs_supply_voltage.items():
                    target_dvfs_supply_voltage[key] = kwargs['target_dvfs_supply_voltage_constant']
        else:
            target_dvfs_supply_voltage = None

        if ('target_dvfs_idle_power_path' in kwargs.keys()) and (kwargs['target_dvfs_idle_power_path'] is not None):
            with open(kwargs['target_dvfs_idle_power_path'], 'r') as f:
                target_dvfs_idle_power = json.load(f)
        else:
            target_dvfs_idle_power = None
                
        # query list: [(query, op_type), (query, op_type), ...]

        estimated_time = 0
        estimated_energy = 0

        # unique workloads in the query_list + counts
        def make_hashable(item):
            dict_part, list_part = item
            # Convert dict to frozenset and list to tuple
            return (frozenset(dict_part.items()), tuple(list_part))

        _dict = [make_hashable(item) for item in query_list]
        counts = Counter(_dict)

        unique_workloads = []
        for (frozen_dict, tuple_list), count in counts.items():
            original_dict = dict(frozen_dict)
            original_list = list(tuple_list)
            unique_workloads.append((original_dict, original_list, count))

        for (q, op, c) in unique_workloads:
            (t, e) = self.predict_single_kernel(q, tuple(op), target_freq=target_freq, verbose=verbose, \
                                                target_gpu_config=target_gpu_config, \
                                                target_dvfs_supply_voltage=target_dvfs_supply_voltage, \
                                                target_dvfs_idle_power=target_dvfs_idle_power, \
                                                freq_auto_search=kwargs['extrapolation_freq_auto_search'], \
                                                use_precomputed_coeffs=use_precomputed_coeffs)

            if type(t) == pd.core.series.Series:
                t = t.values[0]
            if type(e) == pd.core.series.Series:
                e = e.values[0]

            estimated_time += t * c
            estimated_energy += e * c

            if verbose:
                print("Estimated time: {:.4f} ms".format(t))
                print("Estimated energy: {:.4f} J".format(e))
                print("======================================\n")

        return (estimated_time, estimated_energy)