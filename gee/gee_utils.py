import os
import sys

import warnings
warnings.filterwarnings('ignore')

import yaml
import json

from gee.gpu_energy_estimator import Gee

def get_gee(gpu_yaml_path, lut_yaml_path, \
            use_entire_references_for_estimate=False, \
            dvfs_aware=False, \
            dvfs_inference_mode='single_source', \
            dvfs_supply_voltage_json=None, \
            dvfs_idle_power_json=None, \
            lut_folder_abs_path='/mnt/c/Users/KyungmiLee/Documents/gpu-energy-estimation/lut', \
            kernel_predictor_separate_prefill_decode=False, \
            kernel_predict_with_small_kernels=False, \
            multiple_configs=False, \
            gpu_configs_yaml=None, \
            dvfs_idle_power_configs_yaml=None, \
            dvfs_supply_voltage_configs_yaml=None, \
            no_build=False, \
            random_seed=0):

    if not multiple_configs:
        with open(gpu_yaml_path) as f:
            try:
                gpu_config = yaml.safe_load(f)['gpu_configs']
            except yaml.YAMLError as exc:
                print(exc)
                print("Error while loading gpu configuration yaml file")
                exit()
    
    else:
        gpu_config = None

    with open(lut_yaml_path) as f:
        try:
            lut_config = yaml.safe_load(f)['lut_config']
        except yaml.YAMLError as exc:
            print(exc)
            print("Error while loading gpu configuration yaml file")
            exit()

    # DVFS Json
    if dvfs_aware and (not multiple_configs):
        if (dvfs_supply_voltage_json is None):
            print("Error: DVFS supply voltage json file is invalid!")
            exit()
        with open(dvfs_supply_voltage_json, 'r') as f:
            dvfs_supply_voltage = json.load(f)

        if (dvfs_idle_power_json is None) :
            print("Error: DVFS idle power json file is invalid!")
            exit()
        with open(dvfs_idle_power_json, 'r') as f:
            dvfs_idle_power = json.load(f)
    else:
        dvfs_supply_voltage = None
        dvfs_idle_power = None

    # Multiple configuration - extrapolation cases
    gpu_configs = {}
    dvfs_supply_voltage_configs = {}
    dvfs_idle_power_configs = {}

    if multiple_configs:
        with open(gpu_configs_yaml, 'r') as f:
            gpu_configs_filenames = yaml.safe_load(f)
        for key, value in gpu_configs_filenames.items():
            with open(value, 'r') as f:
                tmp_config = yaml.safe_load(f)['gpu_configs']
            gpu_configs[key] = tmp_config
        
        if dvfs_supply_voltage_configs_yaml is None:
            print("Warning!!! Supply voltage yaml is None!")
        else:
            with open(dvfs_supply_voltage_configs_yaml, 'r') as f:
                voltage_filenames = yaml.safe_load(f)
            for key, value in voltage_filenames.items():
                with open(value, 'r') as f:
                    dvfs_supply_voltage_configs[key] = json.load(f)
        
        if dvfs_idle_power_configs_yaml is None:
            print("Warning!!! Idle power yaml is None!")
        else:
            with open(dvfs_idle_power_configs_yaml, 'r') as f:
                power_filenames = yaml.safe_load(f)
            for key, value in power_filenames.items():
                with open(value, 'r') as f:
                    dvfs_idle_power_configs[key] = json.load(f)

    gpu_energy_estimator = Gee(lut_config=lut_config, \
                               gpu_config=gpu_config, \
                               use_entire_references_for_estimate=use_entire_references_for_estimate, \
                               dvfs_aware=dvfs_aware, \
                               dvfs_inference_mode=dvfs_inference_mode, \
                               dvfs_supply_voltage=dvfs_supply_voltage, \
                               dvfs_idle_power=dvfs_idle_power, \
                               lut_folder_abs_path=lut_folder_abs_path, \
                               kernel_predictor_separate_prefill_decode=kernel_predictor_separate_prefill_decode, \
                               kernel_predict_with_small_kernels=kernel_predict_with_small_kernels, \
                               multiple_configs=multiple_configs, \
                               gpu_configs=gpu_configs, \
                               dvfs_idle_power_configs=dvfs_idle_power_configs, \
                               dvfs_supply_voltage_configs=dvfs_supply_voltage_configs, \
                               no_build=no_build, \
                               random_seed=random_seed)
    
    return gpu_energy_estimator
    
