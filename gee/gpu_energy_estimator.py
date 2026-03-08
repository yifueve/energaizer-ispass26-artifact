import os
import numpy as np
import pandas as pd
import math
import copy
import csv
import json
import re
import yaml
import time

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('..')

from gee.estimator import GemmLikeEstimator, NonlinearEstimator, NcclEstimator, ElementwiseEstimator, FlashAttentionEstimator
from gee.frontend_utils import *

class Gee():
    def __init__(self, lut_config, gpu_config, \
                 dvfs_aware=False, dvfs_inference_mode='single_source', dvfs_supply_voltage={}, dvfs_idle_power={}, \
                 lut_folder_abs_path='/home/kyungmi/gpu-energy-estimation/lut', \
                 use_entire_references_for_estimate=False, \
                 kernel_predictor_separate_prefill_decode=False, kernel_predict_with_small_kernels=False, \
                 multiple_configs=False, gpu_configs={}, dvfs_idle_power_configs={}, dvfs_supply_voltage_configs={}, \
                 no_build=False, random_seed=0):
        
        self.dvfs_aware = dvfs_aware
        self.dvfs_inference_mode = dvfs_inference_mode
        self.dvfs_supply_voltage = dvfs_supply_voltage
        self.dvfs_idle_power = dvfs_idle_power

        # Einsum cache
        self.einsum_parse_cache = {}
        self.enable_einsum_cache = True

        # GEMM estimator kernel prediction option
        self.kernel_predict_with_small_kernels = kernel_predict_with_small_kernels

        # Version
        self.version = 3

        # Estimators
        self.gemm_estimator = GemmLikeEstimator(lut_config, gpu_config, \
                                                dvfs_aware=self.dvfs_aware, dvfs_inference_mode=self.dvfs_inference_mode, \
                                                dvfs_supply_voltage=self.dvfs_supply_voltage, dvfs_idle_power=self.dvfs_idle_power, \
                                                lut_folder_abs_path=lut_folder_abs_path, \
                                                use_entire_references_for_estimate=use_entire_references_for_estimate, conv2d=False, \
                                                kernel_predictor_separate_prefill_decode=kernel_predictor_separate_prefill_decode, \
                                                multiple_configs=multiple_configs, gpu_configs=gpu_configs, \
                                                dvfs_idle_power_configs=dvfs_idle_power_configs, dvfs_supply_voltage_configs=dvfs_supply_voltage_configs, \
                                                random_seed=random_seed)
        self.nonlinear_estimator = NonlinearEstimator(lut_config, gpu_config, \
                                                      dvfs_aware=self.dvfs_aware, dvfs_inference_mode=self.dvfs_inference_mode, \
                                                      dvfs_supply_voltage=self.dvfs_supply_voltage, dvfs_idle_power=self.dvfs_idle_power, \
                                                      lut_folder_abs_path=lut_folder_abs_path, \
                                                      multiple_configs=multiple_configs, gpu_configs=gpu_configs, \
                                                      dvfs_idle_power_configs=dvfs_idle_power_configs, dvfs_supply_voltage_configs=dvfs_supply_voltage_configs)
        self.nccl_estimator = NcclEstimator(lut_config, gpu_config, \
                                            dvfs_aware=self.dvfs_aware, dvfs_inference_mode=self.dvfs_inference_mode, \
                                            dvfs_supply_voltage=self.dvfs_supply_voltage, dvfs_idle_power=self.dvfs_idle_power, \
                                            lut_folder_abs_path=lut_folder_abs_path)
        self.conv_estimator = GemmLikeEstimator(lut_config, gpu_config, \
                                                dvfs_aware=self.dvfs_aware, dvfs_inference_mode=self.dvfs_inference_mode, \
                                                dvfs_supply_voltage=self.dvfs_supply_voltage, dvfs_idle_power=self.dvfs_idle_power, \
                                                lut_folder_abs_path=lut_folder_abs_path, \
                                                use_entire_references_for_estimate=use_entire_references_for_estimate, conv2d=True, \
                                                multiple_configs=multiple_configs, gpu_configs=gpu_configs, \
                                                dvfs_idle_power_configs=dvfs_idle_power_configs, dvfs_supply_voltage_configs=dvfs_supply_voltage_configs)
        self.elementwise_estimator = ElementwiseEstimator(lut_config, gpu_config, \
                                                          dvfs_aware=self.dvfs_aware, dvfs_inference_mode=self.dvfs_inference_mode, \
                                                          dvfs_supply_voltage=self.dvfs_supply_voltage, dvfs_idle_power=self.dvfs_idle_power, \
                                                          lut_folder_abs_path=lut_folder_abs_path, \
                                                          multiple_configs=multiple_configs, gpu_configs=gpu_configs, \
                                                          dvfs_idle_power_configs=dvfs_idle_power_configs, dvfs_supply_voltage_configs=dvfs_supply_voltage_configs)
        self.flashattn_estimator = FlashAttentionEstimator(lut_config, gpu_config, \
                                                           dvfs_aware=self.dvfs_aware, dvfs_inference_mode=self.dvfs_inference_mode, \
                                                           dvfs_supply_voltage=self.dvfs_supply_voltage, dvfs_idle_power=self.dvfs_idle_power, \
                                                           lut_folder_abs_path=lut_folder_abs_path, \
                                                           multiple_configs=multiple_configs, gpu_configs=gpu_configs, \
                                                           dvfs_idle_power_configs=dvfs_idle_power_configs, dvfs_supply_voltage_configs=dvfs_supply_voltage_configs)

        # Build
        self.gemm_estimator.build(no_build)
        self.nonlinear_estimator.build(no_build)
        self.conv_estimator.build(no_build)
        self.elementwise_estimator.build(no_build)
        self.flashattn_estimator.build()

    def __version__(self):
        return self.version
    
    def save_prepared_database(self):
        self.gemm_estimator.save_prepared_database()
    
    def empty_einsum_cache(self):
        self.einsum_parse_cache = {}
        return

    def lookup(self, query, query_type, target_freq=None, verbose=False, lookup_target='energy', \
               target_gpu_config=None, target_dvfs_supply_voltage=None, target_dvfs_idle_power=None, \
               kernel_info_provided=None, return_power_flag=False, use_precomputed_coeff=False):
        assert (lookup_target in ['energy', 'time', 'power', 'all'])

        # Previous version compatability
        if type(query_type) == str:
            query_type = (query_type, )

        if query_type[0] == 'tc_gemm':
            query_type = ('gemm', 'tc', query_type[1])
        elif query_type[0] == 'cuda_gemm':
            query_type = ('gemm', 'cuda', query_type[1])
        
        # Check which estimator to call
        if query_type[0] in self.gemm_estimator.ops_supported:
            estimator = self.gemm_estimator
        elif query_type[0] in self.nonlinear_estimator.ops_supported:
            estimator = self.nonlinear_estimator
        elif query_type[0] == 'nccl':
            estimator = self.nccl_estimator
        elif query_type[0] in self.conv_estimator.ops_supported:
            estimator = self.conv_estimator
        elif query_type[0] == 'elementwise':
            estimator = self.elementwise_estimator
        elif query_type[0] == 'flashattention_v2':
            estimator = self.flashattn_estimator
        else:
            raise NotImplementedError('Query type of {} is not supported!'.format(query_type[0]))
    
        if (kernel_info_provided is not None) and (query_type[0] == 'gemm'):
            try:
                (estimated_time, estimated_power, estimated_energy) = estimator.predict(query, query_type[0], target_freq, verbose=verbose, \
                                                                                        target_gpu_config=target_gpu_config, target_dvfs_supply_voltage=target_dvfs_supply_voltage, \
                                                                                        target_dvfs_idle_power=target_dvfs_idle_power, \
                                                                                        kernel_info_provided=kernel_info_provided, \
                                                                                        condition_subset=['block_tile_M', 'block_tile_N', 'block_tile_K', \
                                                                                                        'warp_tile_M', 'warp_tile_N', 'warp_tile_K', \
                                                                                                        'math_inst_M', 'math_inst_N', 'math_inst_K', \
                                                                                                        'gemv', 'use_cuda_core_only', 'multistageK', 'threads'])
            except:
                print("Using the provided kernel information failed - there might not exist any corresponding entry in the LUT!")
                print("Return the estimated one..")
                (estimated_time, estimated_power, estimated_energy) = estimator.predict(query, query_type[0], target_freq, verbose=verbose, \
                                                                                        target_gpu_config=target_gpu_config, target_dvfs_supply_voltage=target_dvfs_supply_voltage, \
                                                                                        target_dvfs_idle_power=target_dvfs_idle_power, predict_with_smaller_kernels=self.kernel_predict_with_small_kernels)

        else:
            # Run estimator
            estimated = estimator.predict(query, query_type[0], target_freq, verbose=verbose, \
                                          target_gpu_config=target_gpu_config, target_dvfs_supply_voltage=target_dvfs_supply_voltage, \
                                          target_dvfs_idle_power=target_dvfs_idle_power, \
                                          return_power_flag=return_power_flag, predict_with_smaller_kernels=self.kernel_predict_with_small_kernels, \
                                          use_precomputed_coeffs=use_precomputed_coeff)
            estimated_time = estimated[0]
            estimated_power = estimated[1]
            estimated_energy = estimated[2]

            if (query_type[0] == 'gemm') and return_power_flag:
                estimation_flag = estimated[3]
            else:
                estimation_flag = None

        if type(estimated_energy) == pd.core.series.Series:
            estimated_energy = estimated_energy.values[0]
        if type(estimated_time) == pd.core.series.Series:
            estimated_time = estimated_time.values[0]

        if lookup_target == 'energy':
            return estimated_energy
        
        elif lookup_target == 'time':
            return estimated_time
        
        elif lookup_target == 'power':
            return estimated_power
        
        else:
            if return_power_flag:
                return (estimated_time, estimated_power, estimated_energy, estimation_flag)
            else:
                return (estimated_time, estimated_power, estimated_energy)
        
    def lookup_einsum(self, einsum_args, precM, precA, use_tensorcore, target_freq=None, lookup_target='energy', verbose=False, \
                      kernel_info=None, transpose_mn=False):
        gemm_list = parse_einsum(einsum_args, self.einsum_parse_cache, self.enable_einsum_cache, False)

        if kernel_info is not None:
            assert (len(gemm_list)==1, 'The number of GEMMs inside this Einsum equation is more than 1!')
            if 'max_concurrent_block' not in kernel_info.keys():
                kernel_info['max_concurrent_block'] = 2
            kernel_info['batch'] = gemm_list[0]['matA'][0]
            kernel_info['dimM'] = gemm_list[0]['matA'][1]
            kernel_info['dimN'] = gemm_list[0]['matB'][2]
            kernel_info['dimK'] = gemm_list[0]['matA'][2]
            kernel_info['precM'] = precM
            kernel_info['precA'] = precA
            kernel_info['useTensorCore'] = use_tensorcore
            kernel_info_provided = self.gemm_estimator.kernel_parser.parse(kernel_info, 'gemm')
        else:
            kernel_info_provided = None

        energy_list = []

        for gemm_idx, gemm in enumerate(gemm_list):
            resultvars = {}
            resultvars['batch'] = gemm['matA'][0]
            resultvars['dimM'] = gemm['matA'][1] if not transpose_mn else gemm['matB'][2]
            resultvars['dimN'] = gemm['matB'][2] if not transpose_mn else gemm['matA'][1]
            resultvars['dimK'] = gemm['matA'][2]
            resultvars['precM'] = precM
            resultvars['precA'] = precA
            resultvars['useTensorCore'] = use_tensorcore

            if verbose or self.print_debug_signals:
                print("Resolved to : batch {} | M {} | N {} | K {}".format(resultvars['batch'], resultvars['dimM'], resultvars['dimN'], resultvars['dimK']))
            
            query_type = ('gemm', 'tc' if use_tensorcore else 'cuda', '{}_{}'.format(precM, precA))
            estimated = self.lookup(resultvars, query_type, target_freq, verbose=verbose, lookup_target=lookup_target, kernel_info_provided=kernel_info_provided)

            energy_list.append(estimated)
            
        return energy_list
                
