import os
import sys
import argparse
import copy
import json

import warnings
warnings.filterwarnings('ignore')

from datetime import datetime

import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm

from gee_estimator import GeeEstimator
from limicro_estimator import LiEstimator
from neusight_estimator import NeuSightEstimator
from roofline_estimator import RooflineEstimator

import time 

def main():
    parser = argparse.ArgumentParser()

    # Common
    parser.add_argument('--workload_folder', type=str, required=True)
    parser.add_argument('--result_save_to', type=str, required=True)
    parser.add_argument('--result_filename', type=str, required=False)

    parser.add_argument('--gpu_config_yaml', type=str, required=False)
    parser.add_argument('--lut_config_yaml', type=str, required=True)
    parser.add_argument('--lut_folder_abs_path', type=str, default='/mnt/c/Users/KyungmiLee/Documents/energaizer-ispass-artifact/database/data')

    # Turn off some estimators
    parser.add_argument('--no_gee', default=False, action='store_true')
    parser.add_argument('--no_limicro', default=False, action='store_true')
    parser.add_argument('--no_neusight', default=False, action='store_true')
    parser.add_argument('--no_roofline', default=False, action='store_true')

    # NeuSight Related
    parser.add_argument('--neusight_gemm_train_config', type=str, required=False)
    parser.add_argument('--neusight_nonlinear_train_config', type=str, required=False)
    parser.add_argument('--neusight_pretrained', default=False, action='store_true')
    parser.add_argument('--neusight_pretrained_model_path', default='', type=str, required=False)
    parser.add_argument('--neusight_save_model', default=False, action='store_true')
    parser.add_argument('--neusight_save_to', default='', type=str, required=False)

    # DVFS Related
    parser.add_argument('--dvfs_aware', action='store_true', default=False)
    parser.add_argument('--dvfs_supply_voltage_json', type=str, required=False)
    parser.add_argument('--dvfs_idle_power_json', type=str, required=False)
    parser.add_argument('--dvfs_gee_use_entire_ref', default=False, action='store_true')
    parser.add_argument('--target_freq', type=int, default=900)
    parser.add_argument('--freq_min', type=int, default=510)
    parser.add_argument('--freq_max', type=int, default=1410)
    parser.add_argument('--freq_step', type=int, default=45)

    # Extrapolation Related
    parser.add_argument('--extrapolation', default=False, action='store_true')
    parser.add_argument('--extrapolation_freq_auto_search', default=False, action='store_true')
    parser.add_argument('--target_gpu_config_yaml', type=str, required=False)
    parser.add_argument('--target_supply_voltage_json', type=str, required=False)
    parser.add_argument('--target_supply_voltage_value', type=float, default=-1, required=False)
    parser.add_argument('--target_idle_power_json', type=str, required=False)
    parser.add_argument('--target_freq_align_json', type=str, required=False)

    # Extrapolation - multiple-gpu database
    parser.add_argument('--multiple_config', default=False, action='store_true')
    parser.add_argument('--multiple_gpu_configs_yaml', type=str)
    parser.add_argument('--multiple_gpu_supply_voltage_yaml', type=str)
    parser.add_argument('--multiple_gpu_idle_power_yaml', type=str)

    # Flash Attention - GEMM approximate or separate kernel type
    parser.add_argument('--flash_attention_enable', default=False, action='store_true')
    parser.add_argument('--flash_attention_estimate_method', choices=['fmha-approximate', 'flashattention_v2'], required=False)

    # Kernel predictor split decode/prefill kernels
    parser.add_argument('--kernel_predictor_separate_prefill_decode', default=False, action='store_true')

    # Precomputed coeffs
    parser.add_argument('--gee_use_precomputed_coeffs', default=False, action='store_true')

    args = parser.parse_args()

    print("Testing workloads in the following folder: ", args.workload_folder)

    # Get the summary filename
    workload_folder_name = os.path.basename(os.path.normpath(args.workload_folder))
    current_time = datetime.now()
    timestamp = current_time.strftime("%m%d%y_%H%M%S")

    if not os.path.exists(args.result_save_to):
        os.makedirs(args.result_save_to, exist_ok=True)

    if args.result_filename:
        result_filename = os.path.join(args.result_save_to, args.result_filename)
    else:
        result_filename = os.path.join(args.result_save_to, '{}_{}.csv'.format(workload_folder_name, timestamp))

    print("Result will be save to: ", result_filename)

    estimators = []
    estimator_tag = []

    # Get estimators
    if not args.no_gee:
        gee = GeeEstimator(args.gpu_config_yaml, \
                           args.lut_config_yaml, \
                           args.lut_folder_abs_path, \
                           dvfs_use_entire_references_for_estimate=args.dvfs_gee_use_entire_ref, \
                           dvfs_aware=(args.dvfs_aware or args.extrapolation), \
                           dvfs_inference_mode='all', \
                           dvfs_supply_voltage_json=args.dvfs_supply_voltage_json, \
                           dvfs_idle_power_json=args.dvfs_idle_power_json, \
                           kernel_predictor_separate_prefill_decode=args.kernel_predictor_separate_prefill_decode, \
                           multiple_config=args.multiple_config, \
                           multiple_gpu_configs_yaml=args.multiple_gpu_configs_yaml, \
                           multiple_gpu_idle_power_yaml=args.multiple_gpu_idle_power_yaml, \
                           multiple_gpu_supply_voltage_yaml=args.multiple_gpu_supply_voltage_yaml)
        print("Done building Gee!")

        estimators.append(gee)
        estimator_tag.append('gee')

    if not args.no_neusight:
        neusight = NeuSightEstimator(args.gpu_config_yaml, \
                                     args.lut_config_yaml, \
                                     args.lut_folder_abs_path, \
                                     gemm_train_config=args.neusight_gemm_train_config, \
                                     nonlinear_train_config=args.neusight_nonlinear_train_config, \
                                     pretrained=args.neusight_pretrained, \
                                     pretrained_model_path=args.neusight_pretrained_model_path, \
                                     save_model=args.neusight_save_model, \
                                     save_to=args.neusight_save_to, \
                                     kernel_predictor_separate_prefill_decode=args.kernel_predictor_separate_prefill_decode, \
                                     dvfs_aware=(args.dvfs_aware or args.extrapolation), \
                                     multiple_config=args.multiple_config, \
                                     multiple_gpu_configs_yaml=args.multiple_gpu_configs_yaml, \
                                     multiple_gpu_idle_power_yaml=args.multiple_gpu_idle_power_yaml, \
                                     multiple_gpu_supply_voltage_yaml=args.multiple_gpu_supply_voltage_yaml)
        print("Done building NeuSight!")

        estimators.append(neusight)
        estimator_tag.append('neusight')

    if not args.no_roofline:
        roofline = RooflineEstimator(args.gpu_config_yaml, \
                                    args.lut_config_yaml, \
                                    args.lut_folder_abs_path)
        print("Done building Roofline!")

        estimators.append(roofline)
        estimator_tag.append('roofline')

    if not args.no_limicro:
        limicro = LiEstimator(args.gpu_config_yaml, \
                              args.lut_config_yaml, \
                              args.lut_folder_abs_path)
        print("Done building LiMicro!")

        estimators.append(limicro)
        estimator_tag.append('limicro')

    results = []
    
    if args.dvfs_aware and (not args.extrapolation):
        freqs = list(range(args.freq_min, args.freq_max+1, args.freq_step))
    else:
        freqs = [args.target_freq]
    
    if args.target_freq_align_json is not None:
        with open(args.target_freq_align_json, 'r') as f:
            target_freq_align = json.load(f)
    else:
        target_freq_align = None

    for filename in tqdm(os.listdir(args.workload_folder)):
        file_path = os.path.join(args.workload_folder, filename)
        if not os.path.isfile(file_path):
            continue
        
        with open(file_path, 'r') as f:
            query_list = json.load(f)

        # Flash attention estimation mode
        if args.flash_attention_enable:
            for i, q in enumerate(query_list):
                if q[1][0] == 'sdpa':
                    query_list[i][1][0] = args.flash_attention_estimate_method

                    if args.flash_attention_estimate_method == 'flashattention_v2':
                        query_list[i][0]['prec'] = q[0]['precM']
                        query_list[i][0]['batch'] = q[0]['batch_size']
                        query_list[i][0]['n_head'] = q[0]['num_heads']
                
        if target_freq_align is not None:
            target_freq = target_freq_align[filename]
            freqs = [target_freq]

        for freq in freqs:
            for i, e in enumerate(estimators):

                # Measure wall time
                start_time = time.time()
                time_estimated, energy_estimated = e.predict_model(query_list, freq, verbose=False, \
                                                                   use_precomputed_coeffs=args.gee_use_precomputed_coeffs, \
                                                                   target_gpu_yaml_path=args.target_gpu_config_yaml, \
                                                                   target_dvfs_supply_voltage_path=args.target_supply_voltage_json, \
                                                                   target_dvfs_supply_voltage_constant=args.target_supply_voltage_value, \
                                                                   target_dvfs_idle_power_path=args.target_idle_power_json, \
                                                                   extrapolation_freq_auto_search=args.extrapolation_freq_auto_search)
                end_time = time.time()
                elapsed_time = end_time - start_time
                results.append({'workload': str(filename),  'target_freq': freq, 'estimator': estimator_tag[i], 'time_predicted': time_estimated, 'energy_predicted': energy_estimated, 'walltime': elapsed_time})

    df = pd.DataFrame(results)
    df.to_csv(result_filename, index=False)

if __name__ == '__main__':
    main()