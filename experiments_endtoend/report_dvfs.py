import os
import shutil

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import re
import math
import yaml

import sys
sys.path.append('..')

from scipy.optimize import minimize
from sklearn.utils import shuffle

import time
import copy

from tqdm import tqdm

import argparse

MODELNAME_STRINGS = ['opt-decoder-', 'opt-model-', 'bertmodel_', 'gpt2model_', 'optdecoder_', \
                     'mobilevitmodel_', 'optmodel_', 'resnet101_', 'vitmodel_', 'qwen2model_']

LANGUAGE_MODELS = ['optmodel', 'bertmodel', 'gpt2model', 'optdecoder', 'qwen2model']
VISION_MODELS = ['vitmodel', 'mobilevitmodel', 'resnet101']

def parse_filename(filename, model):
    if model is None:
        model = filename.split('_')[0]

    base = filename.replace('.json', '')
    for s in MODELNAME_STRINGS:
        base = base.replace(s, '')

    
    params = {}
    params['model'] = model

    if model in LANGUAGE_MODELS:
        patterns = {
            'batch': [r'_batch(\d+)_', r'_b(\d+)_'],
            'seq': [r'seq(\d+)', r'_s(\d+)']
        }
    else:
        patterns = {
            'batch': [r'_b(\d+)_']
        }
    
    for key, pattern in patterns.items():
        for p in pattern:
            match = re.search(p, base)
            if match:
                # Convert to int for numeric values
                value = match.group(1)
                params[key] = int(value) if value.isdigit() else value
                break

    if model in LANGUAGE_MODELS:
        params['mode'] = 'decode' if ('decode' in base) else 'prefill'
    
    config = base.split('_')[0]
    params['config'] = config

    params['prec'] = base.split('_')[1][1:]

    return params

def get_compute_kernel_portion(ncu_path, prefix):
    gemm_kernel_strings = ['gemm', 'cutlass', 'gemv', 'largek']
    softmax_kernel_strings = ['softmax']
    layernorm_kernel_strings = ['layer_norm']
    conv2d_kernel_strings = ['gemm', 'xmma', 'cutlass', 'cudnn']
    elementwise_strings = ['elementwise']

    kernel_strings = list(set(gemm_kernel_strings + softmax_kernel_strings + layernorm_kernel_strings + conv2d_kernel_strings + elementwise_strings))

    matching_files = [f for f in os.listdir(ncu_path) if prefix in f]
    if len(matching_files) > 1:
        print("Warning: Multiple ncu files are found for ", prefix)
        
    ncu_file = matching_files[0]
    ncu_df = pd.read_csv(os.path.join(ncu_path, ncu_file))

    cycle_key = 'Elapsed Cycles' if 'Elapsed Cycles' in ncu_df.columns else 'sm__cycles_elapsed.max'
    ncu_df['cycles'] = ncu_df[cycle_key].apply(lambda x: eval(x.replace(',', '')) if type(x)==str else x)
    ncu_df['initialize_kernel'] = ncu_df['kernel_name'].apply(lambda x: ('distribution_elementwise_grid_stride_kernel' in x) and ('curand' in x))

    ncu_df = ncu_df.loc[ncu_df['initialize_kernel'] == False] # Drop data generation for benchmarking purpose - not a part of the workload

    total_cycles = ncu_df['cycles'].sum()

    compute_kernel_time = 0
    compute_kernel_cnt = 0
    for idx, row in ncu_df.iterrows():
        kernel_name = row['kernel_name']
        for s in kernel_strings:
            if s in kernel_name.lower():
                compute_kernel_time += row['cycles']
                compute_kernel_cnt += 1
                break

    gemm_time = 0
    gemm_strings = list(set(gemm_kernel_strings + conv2d_kernel_strings))
    for idx, row in ncu_df.iterrows():
        kernel_name = row['kernel_name']
        for s in gemm_strings:
            if s in kernel_name.lower():
                gemm_time += row['cycles']
                break

    nonlinear_time = 0
    nonlinear_strings = list(set(softmax_kernel_strings + layernorm_kernel_strings))
    for idx, row in ncu_df.iterrows():
        kernel_name = row['kernel_name']
        for s in nonlinear_strings:
            if s in kernel_name.lower():
                nonlinear_time += row['cycles']
                break

    elementwise_time = 0
    for idx, row in ncu_df.iterrows():
        kernel_name = row['kernel_name']
        for s in elementwise_strings:
            if s in kernel_name.lower():
                elementwise_time += row['cycles']
                break

    return (compute_kernel_time / total_cycles, gemm_time, nonlinear_time, elementwise_time, total_cycles)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--estimated_result_path', type=str, required=True)
    parser.add_argument('--measurement_path', type=str, required=True)
    parser.add_argument('--measurement_kernel_breakdown', default=False, action='store_true')
    parser.add_argument('--model', type=str, required=False)
    parser.add_argument('--precision', type=str, choices=['bf16', 'fp32'], required=False)
    parser.add_argument('--report_save_to', type=str, required=True)

    # DVFS related
    parser.add_argument('--freq_min', type=int, required=True)
    parser.add_argument('--freq_max', type=int, required=True)
    parser.add_argument('--freq_step', type=int, default=45, required=False)

    args = parser.parse_args()

    if not os.path.exists(args.report_save_to):
        os.makedirs(args.report_save_to, exist_ok=True)

    # Load the estimation result
    estimated = pd.read_csv(args.estimated_result_path)
    
    # Measurements
    nvml_measured_path = os.path.join(args.measurement_path, 'nvml/nvml_parsed.csv')
    ncu_measured_path = os.path.join(args.measurement_path, 'ncu')
    measured = pd.read_csv(nvml_measured_path)

    # Configurations in the estimation result
    estimated_configurations = list(estimated['workload'].unique())
    estimators_used = list(estimated['estimator'].unique())

    # Frequencies sweep
    freqs = list(range(args.freq_min, args.freq_max+1, args.freq_step))

    summary = []
    for config in estimated_configurations:
        config_params = parse_filename(config, args.model) 

        for f in freqs:

            # find filename in the measurement result
            if config_params['model'] in LANGUAGE_MODELS:
                filename_prefix = '{}_{}_p{}_b{}_s{}_mode{}_attneager_freq{}'.format(config_params['model'], \
                                                                                     config_params['config'], \
                                                                                     args.precision if args.precision is not None else config_params['prec'], \
                                                                                     config_params['batch'], \
                                                                                     config_params['seq'], \
                                                                                     config_params['mode'], \
                                                                                     int(f))
            else:
                filename_prefix = '{}_{}_p{}_b{}_freq{}'.format(config_params['model'], \
                                                                config_params['config'], \
                                                                args.precision if args.precision is not None else config_params['prec'], \
                                                                config_params['batch'], \
                                                                int(f))
                
            try:
                measured_match = measured[measured['workload'].str.contains(filename_prefix)].iloc[0]
            except:
                continue

            result = config_params
            result['freq'] = f
            result['measured_time'] = measured_match['time']
            result['measured_energy'] = measured_match['energy']
            result['temp'] = measured_match['temp']

            if args.measurement_kernel_breakdown:
                (ratio, gemm, nonlinear, elementwise, total) = get_compute_kernel_portion(ncu_measured_path, filename_prefix)
                result['ratio_kernel_of_interest'] = ratio
                result['gemm_cycles'] = gemm
                result['nonlinear_cycles'] = nonlinear
                result['elementwise_cycles'] = elementwise
                result['total_cycles'] = total
                result['kernel_time'] = ratio * measured_match['time']
                result['kernel_energy'] = ratio * measured_match['energy']
            
            for e in estimators_used:
                estimated_match = estimated[(estimated['workload'] == config) & (estimated['estimator'] == e) & (estimated['target_freq'] == f)].iloc[0]
                result['{}_time'.format(e)] = estimated_match['time_predicted'] 
                result['{}_energy'.format(e)] = estimated_match['energy_predicted']

            result['measured_freq'] = measured_match['avg_freq']

            summary.append(copy.deepcopy(result))

    summary = pd.DataFrame(summary)
    for e in estimators_used:
        summary['{}_time_abs_pct_err'.format(e)] = np.abs(summary['{}_time'.format(e)] - summary['measured_time']) / summary['measured_time'] * 100.
        summary['{}_energy_abs_pct_err'.format(e)] = np.abs(summary['{}_energy'.format(e)] - summary['measured_energy']) / summary['measured_energy'] * 100.

        if args.measurement_kernel_breakdown:
            summary['{}_time_kernel_abs_pct_err'.format(e)] = np.abs(summary['{}_time'.format(e)] - summary['kernel_time']) / summary['kernel_time'] * 100
            summary['{}_energy_kernel_abs_pct_err'.format(e)] = np.abs(summary['{}_energy'.format(e)] - summary['kernel_energy']) / summary['kernel_energy'] * 100

    # for e in estimators_used:
    #     time_mape = summary['{}_time_abs_pct_err'.format(e)].mean()
    #     energy_mape = summary['{}_energy_abs_pct_err'.format(e)].mean()
    #     if args.measurement_kernel_breakdown:
    #         kernel_time_mape = summary['{}_time_kernel_abs_pct_err'.format(e)].mean()
    #         kernel_energy_mape = summary['{}_energy_kernel_abs_pct_err'.format(e)].mean()

    #     print("Estimator: {}".format(e))
    #     print("Total Time MAPE: {:.2f} %".format(time_mape))
    #     print("Total Energy MAPE: {:.2f} %".format(energy_mape))
    #     if args.measurement_kernel_breakdown:
    #         print("Kernel Time MAPE: {:.2f} %".format(kernel_time_mape))
    #         print("Kernel Energy MAPE: {:.2f} %".format(kernel_energy_mape))
    #     print("\n")

    result_name = args.estimated_result_path.split('/')[-1]
    summary.to_csv(os.path.join(args.report_save_to, result_name), index=False)

if __name__ == '__main__':
    main()