import os
import sys

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import yaml

sys.path.append('..')
from gee import get_gee

from estimator import Estimator

from tqdm import tqdm

class GeeEstimator(Estimator):
    def __init__(self, gpu_config_yaml, lut_config_yaml, lut_folder_abs_path='/mnt/c/Users/KyungmiLee/Documents/gpu-energy-estimation/lut', **kwargs):
        super().__init__(gpu_config_yaml, lut_config_yaml, lut_folder_abs_path, **kwargs)

        self.gee = get_gee(gpu_yaml_path=gpu_config_yaml, \
                           lut_yaml_path=lut_config_yaml, \
                           use_entire_references_for_estimate=kwargs['dvfs_use_entire_references_for_estimate'], \
                           dvfs_aware=kwargs['dvfs_aware'], \
                           dvfs_inference_mode=kwargs['dvfs_inference_mode'], \
                           dvfs_supply_voltage_json=kwargs['dvfs_supply_voltage_json'], \
                           dvfs_idle_power_json=kwargs['dvfs_idle_power_json'], \
                           kernel_predictor_separate_prefill_decode=kwargs['kernel_predictor_separate_prefill_decode'], \
                           lut_folder_abs_path=lut_folder_abs_path, \
                           multiple_configs=kwargs['multiple_config'], \
                           gpu_configs_yaml=kwargs['multiple_gpu_configs_yaml'], \
                           dvfs_idle_power_configs_yaml=kwargs['multiple_gpu_idle_power_yaml'], \
                           dvfs_supply_voltage_configs_yaml=kwargs['multiple_gpu_supply_voltage_yaml'])
        
        self.dvfs_aware = kwargs['dvfs_aware']
        
    def predict_single_kernel(self, query, op_type, target_freq, \
                              use_precomputed_coeffs=False, lookup_target='all', verbose=False, **kwargs):

        if not self.dvfs_aware:
            (t, _, e) = self.gee.lookup(query, query_type=op_type, \
                                        target_freq=target_freq, lookup_target=lookup_target, \
                                        use_precomputed_coeff=use_precomputed_coeffs, verbose=verbose)
        else:
            (t, p, e) = self.gee.lookup(query, query_type=op_type, \
                                        target_freq=target_freq, lookup_target=lookup_target, verbose=verbose, \
                                        target_gpu_config=kwargs['target_gpu_config'], \
                                        target_dvfs_idle_power=kwargs['target_dvfs_idle_power'], \
                                        target_dvfs_supply_voltage=kwargs['target_dvfs_supply_voltage'])
            
            if kwargs['freq_auto_search']:
                while (p > kwargs['target_gpu_config']['power_cap']):
                    target_freq -= 15
                    (t, p, e) = self.gee.lookup(query, query_type=op_type, \
                                                target_freq=target_freq, lookup_target=lookup_target, verbose=verbose, \
                                                target_gpu_config=kwargs['target_gpu_config'], \
                                                target_dvfs_idle_power=kwargs['target_dvfs_idle_power'], \
                                                target_dvfs_supply_voltage=kwargs['target_dvfs_supply_voltage'])
            
            e = p * t / 1000. # Uncapped power estimate
        
        return (t, e)
    
