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

from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier

import sys
sys.path.append('..')

from gee.kernel_parser import GemmLikeParser, ConvParser
from gee.analytical_model import GemmLikeAnalyticalModel
from gee.optimization_utils import optimize

from gee.estimator.BaseEstimator import BaseEstimator

from gee.frontend_utils import fit_vector_size

class GemmLikeEstimator(BaseEstimator):
    def __init__(self, lut_config=None, gpu_config=None, \
                 dvfs_aware=False, dvfs_inference_mode='single_source', dvfs_supply_voltage={}, dvfs_idle_power={}, \
                 lut_folder_abs_path='/mnt/c/Users/KyungmiLee/Documents/gpu-energy-estimation/lut', \
                 use_entire_references_for_estimate=False, conv2d=False, kernel_predictor_separate_prefill_decode=False, \
                 multiple_configs=False, gpu_configs={}, dvfs_idle_power_configs={}, dvfs_supply_voltage_configs={}, \
                 random_seed=0):
        
        op = 'gemm_like'
        
        if conv2d:
            ops_supported = ['conv2d']
        else:
            ops_supported = ['gemm', 'fmha-approximate'] # fmha and sdpa are same thing (FIX to one name later)

        super().__init__(op, ops_supported, gpu_config=gpu_config, \
                         dvfs_aware=dvfs_aware, dvfs_inference_mode=dvfs_inference_mode, dvfs_supply_voltage=dvfs_supply_voltage, dvfs_idle_power=dvfs_idle_power, \
                         multiple_configs=multiple_configs, gpu_configs=gpu_configs, dvfs_idle_power_configs=dvfs_idle_power_configs, dvfs_supply_voltage_configs=dvfs_supply_voltage_configs)

        self.conv2d = conv2d

        if conv2d:
            self.kernel_parser = ConvParser()
        else:
            self.kernel_parser = GemmLikeParser()

        if self.multiple_configs:
            self.analytical_models = {}
            for key in self.gpu_configs.keys():
                 self.analytical_models[key] = GemmLikeAnalyticalModel(self.gpu_configs[key], \
                                                                       dvfs_supply_voltage=self.dvfs_supply_voltage_configs[key], \
                                                                       dvfs_idle_power=self.dvfs_idle_power_configs[key])
            self.analytical_model = None
        else:
            self.analytical_model = GemmLikeAnalyticalModel(self.gpu_config, \
                                                            dvfs_supply_voltage=self.dvfs_supply_voltage, dvfs_idle_power=self.dvfs_idle_power)
            self.analytical_models = None

        # From lut_config, obtain database 
        self.kernel_database = {}
        self.kernel_database_prepared_id = {}
        self.kernel_database_prepared = {}

        self.model_database = {}
        self.model_database_main_id = {}
        self.model_database_require_annotation = {}
        self.model_database_path = {}
        self.model_database_prepared_id = {}
        
        # Multiple config case
        self.model_database_config_key = {}

        # Precomputed coeffs
        self.model_coeff_precomputed = {}

        for key in lut_config.keys():
            if key not in ops_supported:
                continue
            for cuda_tc_key in lut_config[key].keys():
                for k, v in lut_config[key][cuda_tc_key].items():
                    self.kernel_database[(key, cuda_tc_key, k)] = []
                    self.kernel_database_prepared_id[(key, cuda_tc_key, k)] = []
                    self.model_database[(key, cuda_tc_key, k)] = []
                    self.model_database_path[(key, cuda_tc_key, k)] = []
                    self.model_database_require_annotation[(key, cuda_tc_key, k)] = []
                    self.model_database_prepared_id[(key, cuda_tc_key, k)] = []

                    if self.multiple_configs:
                        self.model_database_config_key[(key, cuda_tc_key, k)] = []

                    self.model_coeff_precomputed[(key, cuda_tc_key, k)] = []

                    model_cnt = 0
                    kernel_cnt = 0
                    for i, elem in enumerate(v):
                        df = pd.read_csv(os.path.join(lut_folder_abs_path, elem['path']))
                        
                        if elem['main']:
                            self.model_database_main_id[(key, cuda_tc_key, k)] = model_cnt
                        if elem['require_annotation']:
                            self.model_database_require_annotation[(key, cuda_tc_key, k)].append(model_cnt)
                        if elem['prepared']:
                            self.model_database_prepared_id[(key, cuda_tc_key, k)].append(model_cnt)
                            self.kernel_database_prepared_id[(key, cuda_tc_key, k)].append(kernel_cnt)
                        if elem['use_for_model']:
                            self.model_database[(key, cuda_tc_key, k)].append(df)
                            # self.model_database_require_annotation[(key, cuda_tc_key, k)].append(os.path.join(lut_folder_abs_path, elem['path']))
                            if self.multiple_configs:
                                self.model_database_config_key[(key, cuda_tc_key, k)].append(elem['gpu_config_key'])
                            model_cnt += 1
                        if elem['use_for_kernel']:
                            self.kernel_database[(key, cuda_tc_key, k)].append(df)
                            kernel_cnt += 1
                        self.model_database_path[(key, cuda_tc_key, k)].append(os.path.join(lut_folder_abs_path, elem['path']))

                        if ('precomputed_coeff' in elem.keys()) and elem['precomputed_coeff']:
                            coeffs = pd.read_csv(elem['precomputed_coeff_path'])
                            self.model_coeff_precomputed[(key, cuda_tc_key, k)].append(coeffs)

        for key, df_arr in self.kernel_database.items():
            self.kernel_database[key] = pd.concat(df_arr)
            prepared = True
            for i, df in enumerate(df_arr):
                prepared = prepared and (i in self.kernel_database_prepared_id[key])
            self.kernel_database_prepared[key] = prepared
        
        self.gpu_roofline_ridge = {}
        self._build_roofline_stat()

        self.use_entire_references_for_estimate = use_entire_references_for_estimate
        
        self.kernel_predictor_separate_prefill_decode = kernel_predictor_separate_prefill_decode

        self.random_seed = random_seed

    def _build_roofline_stat(self):
        for kkey, df_arr in self.model_database.items():
            for i, df in enumerate(df_arr):
                if self.multiple_configs:
                    config_key = self.model_database_config_key[kkey][i]
                    gpu_config = self.gpu_configs[config_key]
                else:
                    gpu_config = self.gpu_config

                gpu_hbm_bw = gpu_config['dram_bw'] # gpu_hbm_bw # GB/s
                gpu_hbm_freq = gpu_config['dram_freq'] # MHz
                gpu_sm_flops = {} # TFLOPS / Dictionary
                for key, value in gpu_config.items():
                    if 'flops' in key:
                        tc, prec, _ = key.split('_')
                        gpu_sm_flops['{}_{}'.format(prec, tc=='tc')] = value
                gpu_sm_freq = gpu_config['sm_max_freq'] # MHz

                gpu_hbm_byte_per_cycle = gpu_hbm_bw / (gpu_hbm_freq / 1000.)

                gpu_flop_per_cycle = {}
                gpu_roofline_ridge = {}
                for key, value in gpu_sm_flops.items():
                    gpu_flop_per_cycle[key] = value / (gpu_sm_freq / 1e6)
                    gpu_roofline_ridge[key] = gpu_flop_per_cycle[key] / gpu_hbm_byte_per_cycle

                if not self.multiple_configs:
                    self.gpu_roofline_ridge = gpu_roofline_ridge

                df['byteM'] = df['precM'].apply(lambda x: prec_to_precision_bits(x) / 8)
                df['byteA'] = df['precA'].apply(lambda x: prec_to_precision_bits(x) / 8)
                df['mem_footprint'] = df['batch'] * (df['dimM'] * df['dimN'] * df['byteA'] + df['dimM'] * df['dimK'] * df['byteM'] + df['dimN'] * df['dimK'] * df['byteM'])
                df['flop'] = df['batch'] * df['dimM'] * df['dimN'] * df['dimK'] * 2

    def save_prepared_database(self):
        for key, df_arr in self.model_database.items():
            for i, df in enumerate(df_arr):
                p = self.model_database_path[key][i]
                p = p[:-4] + '_prepared.csv'
                df.to_csv(p, index=False)

    def build(self, no_build_model=False):
        # Kernel database
        for key, df in self.kernel_database.items():
            if self.kernel_database_prepared[key]:
                continue
            self.kernel_parser.parse_dataframe(df, key[0])

        # Kernel predictor
        self.build_kernel_predictor()

        if no_build_model:
            return

        # Model database
        for key, df_arr in self.model_database.items():
            for i, df in enumerate(df_arr):
                if i in self.model_database_prepared_id[key]:
                    continue
                if i in self.model_database_require_annotation[key]:
                    self._self_annotate_kernel(df, key[0])
                else:
                    self.kernel_parser.parse_dataframe(df, key[0])
                
                if self.multiple_configs:
                    config_key = self.model_database_config_key[key][i]
                    analytical_model = self.analytical_models[config_key]
                else:
                    analytical_model = self.analytical_model

                analytical_model.model(df, train=True)

    def build_kernel_predictor(self):
        self.kernel_predictor = {}

        self.fields = self._build_kernel_map()

        np.random.seed(self.random_seed)

        def get_idx(row, km):
            tmp = {}
            for k in self.fields:
                tmp[k] = row[k]
            return list(km.keys())[list(km.values()).index(tmp)]
        
        for key, df in self.kernel_database.items():
            km = self._kernel_map[key]
            df['kernel_id'] = df.apply(lambda row: get_idx(row, km), axis=1)
            
            if self.conv2d:
                weak_learner = tree.DecisionTreeClassifier(max_depth=15, ccp_alpha=0.0005)
                kernel_predictor = AdaBoostClassifier(estimator=weak_learner, n_estimators=10)
                kernel_predictor = kernel_predictor.fit(df[['b','m','c','hw','rs','stride','padding']], df['kernel_id'])
            else:
                if self.kernel_predictor_separate_prefill_decode:
                    df['decode'] = df.apply(lambda row: (row['dimM'] < 128) or (row['dimN'] < 128), axis=1)
                    df_decode = df.loc[df['decode'] == True]
                    df_normal = df.loc[df['decode'] == False]

                    if len(df_decode) > 0:
                        weak_learner = tree.DecisionTreeClassifier(max_depth=15, ccp_alpha=0.0005)
                        decode_predictor = AdaBoostClassifier(estimator=weak_learner, n_estimators=10)
                        X = df_decode[['batch', 'dimM', 'dimN', 'dimK']]
                        X = np.log2(X)
                        decode_predictor = decode_predictor.fit(X, df_decode['kernel_id'])
                    else:
                        decode_predictor = None
                        
                        
                    if len(df_normal) > 0:
                        weak_learner = tree.DecisionTreeClassifier(max_depth=15, ccp_alpha=0.0005)
                        normal_predictor = AdaBoostClassifier(estimator=weak_learner, n_estimators=10)
                        X = df_normal[['batch', 'dimM', 'dimN', 'dimK']]
                        X = np.log2(X)
                        normal_predictor = normal_predictor.fit(X, df_normal['kernel_id'])
                    else:
                        normal_predictor = None

                    kernel_predictor = (decode_predictor, normal_predictor)

                else:
                    X = df[['batch', 'dimM', 'dimN', 'dimK']]
                    X = np.log2(X)
                    weak_learner = tree.DecisionTreeClassifier(max_depth=15, ccp_alpha=0.0005)
                    kernel_predictor = AdaBoostClassifier(estimator=weak_learner, n_estimators=10)
                    kernel_predictor = kernel_predictor.fit(X, df['kernel_id'])

            splitk_predictor = tree.DecisionTreeClassifier()
            df['splitK_idx'] = df['splitK'].apply(lambda x: 1 if x else 0)
            df_train_exclude_gemv = df.loc[df['gemv'] == False]
            if len(df_train_exclude_gemv) > 0:

                if self.conv2d:
                    splitk_predictor = splitk_predictor.fit(df_train_exclude_gemv[['b','m','c','hw','rs','stride','padding']], df_train_exclude_gemv['splitK_idx'])
                else:
                    X = df_train_exclude_gemv[['batch', 'dimM', 'dimN', 'dimK']]
                    X = np.log2(X)
                    splitk_predictor = splitk_predictor.fit(X, df_train_exclude_gemv['splitK_idx'])

            self.kernel_predictor[key] = (kernel_predictor, splitk_predictor)

    def _build_kernel_map(self):
        fields = ['block_tile_M', 'block_tile_N', 'block_tile_K', \
                  'warp_tile_M', 'warp_tile_N', 'warp_tile_K', \
                  'math_inst_M', 'math_inst_N', 'math_inst_K', \
                  'gemv', 'use_cuda_core_only', 'multistageK', 'threads', 'max_concurrent_block']
        
        self._kernel_map = {}
        for key, df in self.kernel_database.items():
            unique_combinations = df[fields].drop_duplicates()
            unique_combinations = unique_combinations.to_dict('records')
            km = {idx: value for idx, value in enumerate(unique_combinations)}
            self._kernel_map[key] = km

            # for key, value in km.items():
            #     print(value)
            
            # print(df[fields].value_counts())

        return fields
        
    def _self_annotate_kernel(self, df, op):
        columns = list(df.columns)
        for idx, row in df.iterrows():
            query = row.to_dict()
            self.predict_kernel(query, op)
            for k, v in query.items():
                if (k not in columns) and idx==0:
                    df[k] = 0
                    df[k] = df[k].astype(type(v))
                df.loc[idx, k] = v

    def save_fixed_coeff_map(self, save_to):
        for key, df_arr in self.model_database.items():
            for i, df in enumerate(df_arr):
                lutname = self.model_database_path[key][i].split('/')[-1]
                p = lutname[:-4] + '_coeffs.csv'
                p = os.path.join(save_to, p)

                print("Saving the coefficients to the path: {}".format(p))

                # get the kernelmap
                kernelmap = self._kernel_map[key]

                # get the coeff map
                coeffs = self._build_fixed_coeff_map(df, kernelmap)

                # save it
                coeffs = pd.DataFrame(coeffs)
                coeffs.to_csv(p, index=False)

    def _build_fixed_coeff_map(self, df, kernel_map):
        # 1) Identify the unique kernels in the database
        # 2) For each kernel, calculate the coefficients (lambdas in the perf model and C in the power model - not supporting dvfs yet)
        # 3) Return: a dictionary where key: kernel type, values: {perf: coeffs, power: coeffs}

        coeffs = []
        for _, kernel_field in kernel_map.items():
            coeff_dict = {}
            cond = True
            for f in self.fields:
                cond = cond & (df[f] == kernel_field[f])
                coeff_dict[f] = kernel_field[f]
            ref = df.loc[cond]

            # perf
            try:
                z = ref[['t_start', 't_work', 't_end']].to_numpy()
                y = ref['cycles'].to_numpy()
                lambda_1, lambda_2, lambda_3, lambda_4 = optimize(z, y)
                perf_success = True

            except:
                perf_success = False

            # power
            try:
                self._power_activity_factors(ref, (lambda_1, lambda_2, lambda_3, lambda_4), True, False)
                z = ref[['a_dram', 'a_l2', 'a_smem', 'a_math', 'a_other']].to_numpy()
                y = (1000 * ref['energy'].values / ref['time'].values)
                p_dram, p_l2, p_smem, p_math, p_other, p_static = optimize(z, y)
                power_success = True
            
            except:
                power_success = False

            if (perf_success and power_success):
                coeff_dict['perf_coeffs'] = (lambda_1, lambda_2, lambda_3, lambda_4)
                coeff_dict['power_coeffs'] = (p_dram, p_l2, p_smem, p_math, p_other, p_static)
            else:
                coeff_dict['perf_coeffs'] = -1
                coeff_dict['power_coeffs'] = -1

            coeffs.append(coeff_dict)
        
        return coeffs

    def predict_kernel(self, query, op, kernel_info_provided=None, predict_with_smaller_kernels=False, verbose=False):
        assert(op in self.ops_supported)
        cuda_or_tc = 'tc' if query['useTensorCore'] else 'cuda'
        precM = query['precM']
        precA = query['precA']
        prec_str = '{}_{}'.format(precM, precA)

        kernel_predictor, splitk_predictor = self.kernel_predictor[(op, cuda_or_tc, prec_str)]

        if self.kernel_predictor_separate_prefill_decode:
            decode_predictor = kernel_predictor[0]
            normal_predictor = kernel_predictor[1]

            if decode_predictor is None:
                decode_predictor = normal_predictor
            if normal_predictor is None:
                normal_predictor = decode_predictor

            if (query['dimM'] < 128) or (query['dimN'] < 128):
                kernel_predictor = decode_predictor
            else:
                kernel_predictor = normal_predictor

        if kernel_info_provided is None:
            km = self._kernel_map[(op, cuda_or_tc, prec_str)]

            if self.conv2d:
                # 'b','m','c','hw','rs','stride','padding'
                predicted_kernel_id = kernel_predictor.predict([[query['b'], query['m'], query['c'], query['hw'], query['rs'], query['stride'], query['padding']]])[0]
            else:
                if predict_with_smaller_kernels:
                    _batch = query['batch']
                    _dimM = query['dimM']
                    _dimN = query['dimN']
                    _dimK = query['dimK']
                    _batch, _dimM, _dimN, _dimK = fit_vector_size(_batch, _dimM, _dimN, _dimK)
                    if verbose:
                        print("Using smaller problem shape for kernel prediction: batch={}, dimM={}, dimN={}, dimK={}".format(_batch, _dimM, _dimN, _dimK))
                    predicted_kernel_id = kernel_predictor.predict([[np.log2(_batch), np.log2(_dimM), np.log2(_dimN), np.log2(_dimK)]])[0]
                else:
                    predicted_kernel_id = kernel_predictor.predict([[np.log2(query['batch']), np.log2(query['dimM']), np.log2(query['dimN']), np.log2(query['dimK'])]])[0]
            kernel_info = km[predicted_kernel_id]

            if verbose:
                print("Predicted kernel information ---")

        else:
            assert(list(kernel_info_provided.keys()) == self.fields, 'Provided kernel information misses certain fields.')
            kernel_info = kernel_info_provided

            if verbose:
                print("Using the provided kernel information ---")

        if self.conv2d:
            predicted_splitk = bool(splitk_predictor.predict([[query['b'], query['m'], query['c'], query['hw'], query['rs'], query['stride'], query['padding']]])[0])
        else:
            if predict_with_smaller_kernels:
                _batch = query['batch']
                _dimM = query['dimM']
                _dimN = query['dimN']
                _dimK = query['dimK']
                _batch, _dimM, _dimN, _dimK = fit_vector_size(_batch, _dimM, _dimN, _dimK)
                predicted_splitk = bool(splitk_predictor.predict([[np.log2(_batch), np.log2(_dimM), np.log2(_dimN), np.log2(_dimK)]])[0])
            else:
                predicted_splitk = bool(splitk_predictor.predict([[np.log2(query['batch']), np.log2(query['dimM']), np.log2(query['dimN']), np.log2(query['dimK'])]])[0])
            
        if verbose:
            for k, v in kernel_info.items():
                print("{} : {}".format(k, v))
            print("SplitK? {}".format(predicted_splitk))
            print('---------------------------------')

        for k, v in kernel_info.items():
            query[k] = v
        
        # predict the batch of splitk
        lut = self.kernel_database[(op, cuda_or_tc, prec_str)]
        query = self.kernel_parser.calculate_kernel_info_from_prediction(query, kernel_info, predicted_splitk, lut)

    def get_references(self, query, op, condition_exact_match=True, condition_subset=[], ignore_exact=False, min_entries=2, max_entries=20):
        assert(op in self.ops_supported)
        cuda_or_tc = 'tc' if query['useTensorCore'] else 'cuda'
        precM = query['precM']
        precA = query['precA']
        prec_str = '{}_{}'.format(precM, precA)

        lut = self.model_database[(op, cuda_or_tc, prec_str)][self.model_database_main_id[(op, cuda_or_tc, prec_str)]]

        exact_match_found = False
        if not ignore_exact:
            exact = lut.loc[(lut['batch'] == query['batch']) & \
                            (lut['dimM'] == query['dimM']) & \
                            (lut['dimN'] == query['dimN']) & \
                            (lut['dimK'] == query['dimK'])]
            if len(exact) > 0:
                flag = 'exact'
                ref = exact
                exact_match_found = True

        if not exact_match_found:
            cond = True
            for k in condition_subset:
                cond = cond & (lut[k] == query[k])
            ref = lut.loc[cond]
            if (len(ref) < min_entries) and not condition_exact_match:
                ref = self._get_closest_entries(query, lut, condition_subset)

            # if query['batch'] > 1:
            #     _ref = ref.loc[ref['batch'] > 1]
            # else:
            #     _ref = ref.loc[ref['batch'] == 1]
            # if len(_ref) > 0:
            #     ref = _ref

            if query['dimM'] < query['block_tile_M']:
                _ref = ref.loc[ref['dimM'] < query['block_tile_M']]
            else:
                _ref = ref.loc[ref['dimM'] >= query['block_tile_M']]
            if len(_ref) > 0:
                ref = _ref

            if query['dimN'] < query['block_tile_N']:
                _ref = ref.loc[ref['dimN'] < query['block_tile_N']]
            else:
                _ref = ref.loc[ref['dimN'] >= query['block_tile_N']]
            if len(_ref) > 0:
                ref = _ref

            if len(ref) < min_entries:
                flag = 'single'
            else:
                flag = 'ref'
        
        # Trim reference
        ref_copy_before_trim = copy.deepcopy(ref)
        if (flag == 'ref') and (not self.use_entire_references_for_estimate):
            if query['many_blocks']:
                ref = self._get_closest_entries(query, ref_copy_before_trim, ['num_block_tile_K'], exact_condition_subset=['many_blocks', 'many_K'], max_entries=max_entries)
                if len(ref) < min_entries:
                    ref = self._get_closest_entries(query, ref_copy_before_trim, ['num_block_tile_K', 'block_waves'])
            else:
                ref = self._get_closest_entries(query, ref_copy_before_trim, ['block_waves'], exact_condition_subset=['many_blocks', 'many_K'], max_entries=max_entries)
                if len(ref) < min_entries:
                    ref = self._get_closest_entries(query, ref_copy_before_trim, ['block_waves', 'num_block_tile_K'])

            if (len(ref) > max_entries) and (len(ref['block_waves'].unique()) > 1) and (len(ref['num_block_tile_K'].unique()) > 1):
                entries = max_entries
                ref_copy = copy.deepcopy(ref)
                while entries < len(ref_copy):
                    _ref = self._trim_ref(query, ref_copy, max_entries)
                    if (len(_ref['block_waves'].unique()) > 1) and (len(_ref['num_block_tile_K'].unique()) > 1):
                        ref = _ref
                        del ref_copy
                        break
                    entries = entries + math.ceil((len(ref_copy) - entries) / 2)
        
        del ref_copy_before_trim

        return flag, ref

    def _get_closest_entries(self, query, lut, condition_subset, exact_condition_subset=[], max_entries=-1):
        ref = lut
        if len(exact_condition_subset) > 0:
            cond = True
            for k in exact_condition_subset:
                cond = cond & (ref[k] == query[k])
            ref = ref.loc[cond]

        cond_diff_list = []
        for k in condition_subset:
            ref['{}_diff'.format(k)] = np.abs(ref[k] - query[k])
            cond_diff_list.append('{}_diff'.format(k))
        ref.sort_values(by=cond_diff_list, ascending=[True]*len(cond_diff_list), inplace=True)
        ref.reset_index(inplace=True, drop=True)
        if max_entries > 0:
            ref = ref[:max_entries]
        return ref

    def _trim_ref(self, query, ref, max_entries):
        abs_block_diff = np.abs(query['total_block_tiles'] - ref['total_block_tiles'].values) / query['total_block_tiles'] + \
                         np.abs(query['num_block_tile_K'] - ref['num_block_tile_K'].values)  / query['num_block_tile_K']
        indices_to_keep = np.argpartition(abs_block_diff, max_entries)[:max_entries]
        _ref = ref.iloc[indices_to_keep]
        return _ref
    
    def _single_entry_interpolation(self, query, ref, return_power_flag=False):
        # flops = query['batch'] * query['dimM'] * query['dimN'] * query['dimK'] * 2
        # mem = query['batch'] * (query['dimM'] * query['dimN'] * prec_to_precision_bits(query['precA']) + (query['dimM'] * query['dimK'] + query['dimN'] * query['dimK']) * prec_to_precision_bits(query['precM']))
        # mem /= 8
        # comp_intensity = flops / mem

        # roofline_ridge = self.gpu_roofline_ridge['{}_{}'.format(query['precM'], query['useTensorCore'])]
        # if comp_intensity < roofline_ridge:
        #     reference_cycle = ref.iloc[0]['mem_footprint']
        #     query_cycle = mem
        # else:
        #     reference_cycle = ref.iloc[0]['flop']
        #     query_cycle = flops

        # estimated_time = query_cycle / reference_cycle * ref.iloc[0]['time']
        # estimated_energy = query_cycle / reference_cycle * ref.iloc[0]['energy']

        # print(query_cycle/reference_cycle, ref.iloc[0]['time'])

        waves_ratio = query['n_waves'] / ref['n_waves']
        stages_ratio = query['stagesK'] / ref['stagesK']

        estimated_time = ref.iloc[0]['time'] * (waves_ratio * stages_ratio)
        estimated_energy = ref.iloc[0]['energy'] * (waves_ratio * stages_ratio)

        if return_power_flag:
            return (estimated_time, estimated_energy / estimated_time * 1000., estimated_energy, False)
        else:
            return (estimated_time, estimated_energy / estimated_time * 1000., estimated_energy)
    
    def _predict_time(self, query, ref, target_freq, time_coeff_correction, use_precomputed_coeffs=False, precomputed_coeffs=None, verbose=False):

        # Arch-aware model
        try:
            if not use_precomputed_coeffs:
                z = ref[['t_start', 't_work', 't_end']].to_numpy()
                y = ref['cycles'].to_numpy()

                lambda_1, lambda_2, lambda_3, lambda_4 = optimize(z, y)
            else:
                lambdas = precomputed_coeffs[0]
                if type(lambdas) == str:
                    lambdas = eval(lambdas[1:-1])
                (lambda_1, lambda_2, lambda_3, lambda_4) = lambdas

            if len(time_coeff_correction) == 4:
                _lambda_1 = lambda_1 * time_coeff_correction[0]
                _lambda_2 = lambda_2 * time_coeff_correction[1]
                _lambda_3 = lambda_3 * time_coeff_correction[2]
                _lambda_4 = lambda_4 * time_coeff_correction[3]
            else:
                _lambda_1 = lambda_1
                _lambda_2 = lambda_2
                _lambda_3 = lambda_3
                _lambda_4 = lambda_4
            cycles_estimate = query['t_start'] * _lambda_1 + query['t_work'] * _lambda_2 + query['t_end'] * _lambda_3 + _lambda_4
            estimated_time = cycles_estimate / (target_freq * 10**3)
            coeffs = (lambda_1, lambda_2, lambda_3, lambda_4)

            if verbose:
                print("Architecture aware modeling success!")
                # print(ref.to_markdown())
                print("Time estimate coefficients ---")
                print("Lambda1 (start): {:.4f} | Lambda2 (work): {:.4f} | Lambda3 (end): {:.4f} | Lambda4 (const): {:.4f}".format(lambda_1, lambda_2, lambda_3, lambda_4))

        # Fall-back to simple linear regression
        except Exception as e:
            if verbose:
                import traceback
                traceback.print_exc()
                print(e)
            ref_copy = copy.deepcopy(ref)
            unique_k = ref['num_block_tile_K'].unique()
            k_match = True
            if (len(unique_k) > 1) or (unique_k[0] != query['num_block_tile_K']):
                k_match = False
            if not k_match:
                ref_copy['y'] = ref_copy['time'] / ref_copy['block_waves']
                ref_copy['x'] = ref_copy['num_block_tile_K']
                t_stage, t_fixed = optimize(ref_copy[['x']].to_numpy(), ref_copy['y'].to_numpy())
                ref_copy['time'] = ref_copy.apply(lambda row: row['block_waves'] * (t_fixed + t_stage * query['num_block_tile_K']) \
                                                if row['num_block_tile_K'] != query['num_block_tile_K'] else row['time'], axis=1)
            avg_time_per_block = (ref_copy['time'] / ref_copy['block_waves']).mean()
            estimated_time = avg_time_per_block * query['block_waves']
            coeffs = None
            del ref_copy

            if verbose:
                print("Architecture aware modeling failed!")
                print("Fall back to a naive linear regression based on waves ---")

        return estimated_time, coeffs
    
    def _power_activity_factors(self, df, lambdas, train=True, \
                                dvfs=False):
        lambda_1, lambda_2, lambda_3, lambda_4 = lambdas

        df['time_estimated_corrected'] = lambda_1 * df['t_start'] + lambda_2 * df['t_work'] + lambda_3 * df['t_end'] + lambda_4
        df['weight_start_full_capacity'] = lambda_1 * df['t_start_per_wave_full_capacity'] * (df['n_waves'] - 1) / df['time_estimated_corrected']
        df['weight_start_last_wave'] = lambda_1 * df['t_start_per_wave_last_wave'] / df['time_estimated_corrected']
        df['weight_work_full_capacity'] = lambda_2 * df['t_work_per_wave_full_capacity'] * (df['n_waves'] - 1) / df['time_estimated_corrected']
        df['weight_work_last_wave'] = lambda_2 * df['t_work_per_wave_last_wave'] / df['time_estimated_corrected']
        df['weight_end_full_capacity'] = lambda_3 * df['t_end_per_wave_full_capacity'] * (df['n_waves'] - 1) / df['time_estimated_corrected']
        df['weight_end_last_wave'] = lambda_3 * df['t_end_per_wave_last_wave'] / df['time_estimated_corrected']
    
        df['a_dram'] = df['weight_start_full_capacity'] * df['a_dram_start_full_capacity'] + \
                    df['weight_start_last_wave'] * df['a_dram_start_last_wave'] + \
                    df['weight_work_full_capacity'] * df['a_dram_work_full_capacity'] + \
                    df['weight_work_last_wave'] * df['a_dram_work_last_wave'] + \
                    df['weight_end_full_capacity'] * df['a_dram_end_full_capacity'] + \
                    df['weight_end_last_wave'] * df['a_dram_end_last_wave']
        df['a_l2'] = df['weight_start_full_capacity'] * df['a_l2_start_full_capacity'] + \
                    df['weight_start_last_wave'] * df['a_l2_start_last_wave'] + \
                    df['weight_work_full_capacity'] * df['a_l2_work_full_capacity'] + \
                    df['weight_work_last_wave'] * df['a_l2_work_last_wave'] + \
                    df['weight_end_full_capacity'] * df['a_l2_end_full_capacity'] + \
                    df['weight_end_last_wave'] * df['a_l2_end_last_wave']
        df['a_smem'] = df['weight_start_full_capacity'] * df['a_smem_start_full_capacity'] + \
                    df['weight_start_last_wave'] * df['a_smem_start_last_wave'] + \
                    df['weight_work_full_capacity'] * df['a_smem_work_full_capacity'] + \
                    df['weight_work_last_wave'] * df['a_smem_work_last_wave'] + \
                    df['weight_end_full_capacity'] * df['a_smem_end_full_capacity'] + \
                    df['weight_end_last_wave'] * df['a_smem_end_last_wave']
        df['a_math'] = df['weight_start_full_capacity'] * df['a_math_start_full_capacity'] + \
                    df['weight_start_last_wave'] * df['a_math_start_last_wave'] + \
                    df['weight_work_full_capacity'] * df['a_math_work_full_capacity'] + \
                    df['weight_work_last_wave'] * df['a_math_work_last_wave'] + \
                    df['weight_end_full_capacity'] * df['a_math_end_full_capacity'] + \
                    df['weight_end_last_wave'] * df['a_math_end_last_wave']
        df['a_other'] = lambda_4 / df['time_estimated_corrected']

        if train:
            df['power'] = (df['energy'] / df['time']) * 1000. 

        if dvfs:
            # df['gpu_supply_voltage'] = df['avg_freq'].apply(lambda x: dvfs_supply_voltage[str(int((x - 210) / 15) * 15 + 210)]) # mV
            # max_voltage = self.dvfs_max_core_voltage # max(dvfs_supply_voltage.values())
            # df['gpu_supply_voltage_norm'] = df['gpu_supply_voltage'].apply(lambda x: x / max_voltage)
            # max_freq = self.dvfs_max_core_freq # max([float(x) for x in dvfs_supply_voltage.keys()])
            # df['gpu_freq_norm'] = df['avg_freq'].apply(lambda x: x / max_freq)
            
            # df['dvfs_scale_factor'] = df['gpu_supply_voltage_norm'] ** 2 * df['gpu_freq_norm'] # V^2 * f
            df['a_l2'] = df['a_l2'] * df['dvfs_scale_factor']
            df['a_smem'] = df['a_smem'] * df['dvfs_scale_factor']
            df['a_math'] = df['a_math'] * df['dvfs_scale_factor']
            df['a_other'] = df['a_other'] * df['dvfs_scale_factor']

            df['extrapolation_sm_scale_factor'] = df['extrapolation_sm_scale_factor'].apply(lambda x: x if x > 0 else 1)
            df['extrapolation_tc_scale_factor'] = df['extrapolation_tc_scale_factor'].apply(lambda x: x if x > 0 else 1)
            df['a_smem'] = df['a_smem'] * df['extrapolation_sm_scale_factor']
            df['a_math'] = df['a_math'] * df['extrapolation_sm_scale_factor'] * df['extrapolation_tc_scale_factor']

            # df['hbm_freq_norm'] = df['hbm_freq'] / self.dvfs_max_hbm_freq
            df['a_dram'] = df['a_dram'] * df['hbm_freq_norm']

            if train:
                # df['power'] = (df['energy'] / df['time']) * 1000. 
                # df['idle_power'] = df['avg_freq'].apply(lambda x: dvfs_idle_power[str(int((x - 210) / 15) * 15 + 210)]) # W
                df['dynamic_power'] = df['power'] - df['idle_power']
                

    def _predict_power(self, query, ref, lambdas, time_coeff_correction=[], \
                       dvfs=False, use_precomputed_coeffs=False, precomputed_coeffs=None, verbose=False):
        
        try:
            query = pd.DataFrame([query], index=[0])
            _lambdas = list(copy.deepcopy(lambdas))
            if len(time_coeff_correction) == len(lambdas):
                for _i in range(len(lambdas)):
                    _lambdas[_i] *= time_coeff_correction[_i]
            _lambdas = tuple(_lambdas)
            self._power_activity_factors(query, _lambdas, False, dvfs)
            query = dict(query.loc[0])

            if not use_precomputed_coeffs:
                self._power_activity_factors(ref, lambdas, True, dvfs)
                # print(ref.to_markdown())
                z = ref[['a_dram', 'a_l2', 'a_smem', 'a_math', 'a_other']].to_numpy()

                if not dvfs:
                    y = (1000 * ref['energy'].values / ref['time'].values)
                    p_dram, p_l2, p_smem, p_math, p_other, p_static = optimize(z, y)
                else:
                    y = ref['dynamic_power'].to_numpy()
                    p_dram, p_l2, p_smem, p_math, p_other = optimize(z, y, const=False)
            else:
                assert(dvfs == False)
                coeffs = precomputed_coeffs[1]
                if type(coeffs) == str:
                    coeffs = eval(coeffs[1:-1])
                (p_dram, p_l2, p_smem, p_math, p_other, p_static) = coeffs

            if not dvfs:
                power_estimate = p_dram * query['a_dram'] + p_l2 * query['a_l2'] + p_smem * query['a_smem'] + p_math * query['a_math'] + p_other * query['a_other'] + p_static
            else:
                power_estimate = p_dram * query['a_dram'] + p_l2 * query['a_l2'] + p_smem * query['a_smem'] + p_math * query['a_math'] + p_other * query['a_other'] 
                power_estimate += query['idle_power']

            if verbose:
                print("Activity factor based power estimation success!")
                if not dvfs:
                    print("Not DVFS! Coefficients ---")
                    print("DRAM: {:.4f} | L2: {:.4f} | Smem: {:.4f} | Math: {:.4f} | Other: {:.4f} | Static: {:.4f}".format(p_dram, p_l2, p_smem, p_math, p_other, p_static))
                else:
                    print("DVFS Aware! Coefficients ---")
                    print("DRAM: {:.4f} | L2: {:.4f} | Smem: {:.4f} | Math: {:.4f} | Other: {:.4f}".format(p_dram, p_l2, p_smem, p_math, p_other))

                print(query)
                print("DRAM: {:.2f} W".format(p_dram * query['a_dram']))
                print("L2: {:.2f} W".format(p_l2 * query['a_l2']))
                print("Shared: {:.2f} W".format(p_smem * query['a_smem']))
                print("Math: {:.2f} W".format(p_math * query['a_math']))
                print("Others: {:.2f} W".format(p_other * query['a_other']))
                
                # print("Lambdas after correction: ", _lambdas)
                # print("Original Lambdas ", lambdas)
                # print("Coeff corrections ", time_coeff_correction)

            
                ref['reconstructed_power'] = p_dram * ref['a_dram'] + p_l2 * ref['a_l2'] + p_smem * ref['a_smem'] + p_math * ref['a_math'] + p_other * query['a_other']
                if dvfs:
                    ref['reconstructed_power'] += query['idle_power']
                else:
                    ref['reconstructed_power'] += p_static

                ref['residual_power'] = ref['power'] - ref['reconstructed_power']

                print(ref.to_markdown())
                # print(query)

            success = True

        except Exception as e:
            if verbose:
                print(e)

            ref_copy = copy.deepcopy(ref)
            ref_copy['power'] = ref_copy['energy'] / ref_copy['time'] * 1000.
            ref_copy['sm_busy'] = (ref_copy['total_block_tiles'] - 1) % ref_copy['gpu_sm'] + 1
            ref_copy['sm_lazy'] = ref_copy['gpu_sm'] - ref_copy['sm_busy']
            ref_copy['x'] = ref_copy['sm_lazy'] * (1 - 1 / ref_copy['block_waves']) + ref_copy['sm_busy']
            slope, intercept = optimize(ref_copy[['x']].to_numpy(), ref_copy['power'].to_numpy())
            sm_busy = (query['total_block_tiles'] - 1) % query['gpu_sm'] + 1
            sm_lazy = query['gpu_sm'] - sm_busy
            power_estimate = intercept + slope * (sm_lazy * (1 - 1 / query['block_waves']) + sm_busy)
            try:
                power_estimate = power_estimate.iloc[0]
            except:
                pass
            
            del ref_copy

            if verbose:
                print("Activity factor based power estimation failed!")
                print("Fall back to simple linear regression based on active sm counts ---")

            success = False

        return power_estimate, success
    
    def _get_target_freq(self, query, lut):
        flop = query['batch'] * query['dimM'] * query['dimN'] * query['dimK']
        ref_copy = copy.deepcopy(lut)
        ref_copy['flop'] = ref_copy['batch'] * ref_copy['dimM'] * ref_copy['dimN'] * ref_copy['dimK']
        ref_copy['flop_abs_diff'] = ref_copy['flop'].apply(lambda x: abs(flop - x))
        ref_copy.sort_values(by='flop_abs_diff', ascending=True, inplace=True)
        ref_copy.reset_index(drop=True, inplace=True)
        target_freq = ref_copy['avg_freq'].values[0]
        del ref_copy

        return target_freq
    
    def _get_precomputed_coeffs(self, query, op):
        assert(op in self.ops_supported)
        cuda_or_tc = 'tc' if query['useTensorCore'] else 'cuda'
        precM = query['precM']
        precA = query['precA']
        prec_str = '{}_{}'.format(precM, precA)

        coeffs = self.model_coeff_precomputed[(op, cuda_or_tc, prec_str)][self.model_database_main_id[(op, cuda_or_tc, prec_str)]]
        cond = True
        for k in self.fields:
            cond = cond & (coeffs[k] == query[k])
        ref = coeffs.loc[cond]

        perf_coeffs = ref['perf_coeffs'].values[0]
        power_coeffs = ref['power_coeffs'].values[0]
        if type(perf_coeffs) == str:
            if (perf_coeffs == "-1"):
                perf_coeffs = -1
            else: 
                perf_coeffs = eval(perf_coeffs[1:-1])
        if type(power_coeffs) == str:
            if (power_coeffs == "-1"):
                power_coeffs = -1
            else:
                power_coeffs = eval(power_coeffs[1:-1])
        return (perf_coeffs, power_coeffs)

    def _predict_gemm(self, query, op, target_freq, kernel_info_provided=None, \
                      condition_exact_match=True, condition_subset=[], ignore_exact=False, min_entries=2, max_entries=20, \
                      adjust_smem_to_gmem_ratio=1, adjust_gmem_to_smem_ratio=1, adjust_gmem_to_smem_select=None, \
                      target_gpu_config=None, target_dvfs_supply_voltage=None, target_dvfs_idle_power=None, \
                      time_coeffs_corrections=[], \
                      verbose=False, return_power_flag=False, predict_with_smaller_kernels=False, use_precomputed_coeffs=False):
        
        # ignore_exact should be true for dvfs
        ignore_exact = ignore_exact or (self.dvfs_aware == True)

        self.predict_kernel(query, op, kernel_info_provided=kernel_info_provided, predict_with_smaller_kernels=predict_with_smaller_kernels, verbose=verbose)
        if len(condition_subset) == 0:
            condition_subset = self.fields
            # if 'gemv' in condition_subset:
            #     condition_subset.remove('gemv')

        if target_freq is None:
            query['avg_freq'] = 1000 # assume 1GHz
        else:
            query['avg_freq'] = target_freq

        # Annotate query
        query = pd.DataFrame([query], index=[0])
        if target_gpu_config is not None:
            target_gpu_analytical_model = GemmLikeAnalyticalModel(target_gpu_config, \
                                                                  dvfs_supply_voltage=target_dvfs_supply_voltage, \
                                                                  dvfs_idle_power=target_dvfs_idle_power)
            target_gpu_analytical_model.model(query, train=False, \
                                              adjust_gmem_to_smem_ratio=adjust_gmem_to_smem_ratio, \
                                              adjust_smem_to_gmem_ratio=adjust_smem_to_gmem_ratio, \
                                              adjust_gmem_to_smem_select=adjust_gmem_to_smem_select)
            del target_gpu_analytical_model
        else:
            if self.multiple_configs:
                print("Error: multiple GPU configuration in the database is only supported for extrapolation cases!")
                exit()
            self.analytical_model.model(query, train=False, \
                                        adjust_gmem_to_smem_ratio=adjust_gmem_to_smem_ratio, \
                                        adjust_smem_to_gmem_ratio=adjust_smem_to_gmem_ratio, \
                                        adjust_gmem_to_smem_select=adjust_gmem_to_smem_select)
        query = dict(query.loc[0])

        precompute_success = False
        if use_precomputed_coeffs:
            flag = 'precompute'
            precomputed_coeffs = self._get_precomputed_coeffs(query, op)
            ref = None

            if ((precomputed_coeffs[0]) == -1):
                precompute_success = False
            else:
                precompute_success = True

        if (not use_precomputed_coeffs) or (not precompute_success):
            flag, ref = self.get_references(query, op, condition_exact_match, condition_subset, \
                                            ignore_exact, min_entries, max_entries)
            
            # Empty target frequency -> only for non-multiple config cases
            # TODO: Add warning
            if (target_freq is None) and (flag == 'ref'):
                target_freq =  self._get_target_freq(query, ref)
                query['avg_freq'] = target_freq
                query = pd.DataFrame([query], index=[0])
                self.analytical_model.model(query, train=False, \
                                            adjust_gmem_to_smem_ratio=adjust_gmem_to_smem_ratio, \
                                            adjust_smem_to_gmem_ratio=adjust_smem_to_gmem_ratio, \
                                            adjust_gmem_to_smem_select=adjust_gmem_to_smem_select)
                query = dict(query.loc[0])
            
            if verbose:
                print("Reference status: {} | Reference entry size: {}".format(flag, len(ref)))
                # print(ref.to_markdown())
                # print(query['avg_freq'])
                # for k, v in query.items():
                #     print("{}: {}".format(k, v))

            if len(ref) == 0:
                print("Invalid choice of kernel and condition options! No reference found!")
                print(query)
                print(flag)
                print(self._kernel_map[(op, 'tc', 'bf16_bf16')])
                exit()

            # DVFS Inference Mode is all, then use all sources
            if self.dvfs_inference_mode == 'all':
                dfs_ref = []
                cuda_or_tc = 'tc' if query['useTensorCore'] else 'cuda'
                precM = query['precM']
                precA = query['precA']
                prec_str = '{}_{}'.format(precM, precA)
                for dvfs_df in self.model_database[(op, cuda_or_tc, prec_str)]:
                    if 'trans' in dvfs_df.columns:
                        dvfs_df = dvfs_df.loc[dvfs_df['trans']=='nn']
                    _df = pd.merge(dvfs_df, ref, on=['batch', 'dimM', 'dimN', 'dimK'], how='inner', indicator=True, suffixes=[None, 'ref'])
                    _df = _df.loc[:, dvfs_df.columns]
                    dfs_ref.append(_df)
                ref = pd.concat(dfs_ref, ignore_index=True)

                # if verbose:
                #     print(ref.to_markdown())
            
            precomputed_coeffs = None

        if flag == 'exact':
            if return_power_flag:
                return (ref.iloc[0]['time'], -1, ref.iloc[0]['energy'], False)
            else:
                return (ref.iloc[0]['time'], -1, ref.iloc[0]['energy'])
        elif (flag == 'single') and (len(ref) == 1):
            if verbose:
                print(ref.to_markdown())
            return self._single_entry_interpolation(query, ref, return_power_flag)
        else:
            estimated_time, coeffs = self._predict_time(query, ref, target_freq, time_coeff_correction=time_coeffs_corrections, \
                                                        use_precomputed_coeffs=(use_precomputed_coeffs and precompute_success), precomputed_coeffs=precomputed_coeffs, verbose=verbose)
            estimated_power, flag = self._predict_power(query, ref, coeffs, time_coeff_correction=time_coeffs_corrections, dvfs=self.dvfs_aware, \
                                                        use_precomputed_coeffs=(use_precomputed_coeffs and precompute_success), precomputed_coeffs=precomputed_coeffs, verbose=verbose)
            if target_gpu_config is not None:
                power = min(estimated_power, target_gpu_config['power_cap'])
            else:
                power = min(estimated_power, self.gpu_config['power_cap'])
            estimated_energy = power * estimated_time / 1000.

            if return_power_flag:
                return (estimated_time, estimated_power, estimated_energy, flag)
            else:
                return (estimated_time, estimated_power, estimated_energy)
        
    def predict(self, query, op, target_freq, kernel_info_provided=None, \
                condition_exact_match=True, condition_subset=[], ignore_exact=False, min_entries=2, max_entries=20, \
                target_gpu_config=None, target_dvfs_supply_voltage=None, target_dvfs_idle_power=None, \
                verbose=False, return_power_flag=False, predict_with_smaller_kernels=False, use_precomputed_coeffs=False):
        
        if (op == 'gemm') or (op == 'conv2d'):
            # query should have following fields:
            # batch, dimM, dimN, dimK, precM, precA, useTensorCore
            assert(set(['batch', 'dimM', 'dimN', 'dimK', 'precM', 'precA', 'useTensorCore']).issubset(list(query.keys())))
            adjust_smem_to_gmem_ratio=1
            adjust_gmem_to_smem_ratio=1
            adjust_gmem_to_smem_select=None
            return self._predict_gemm(query, op, target_freq, kernel_info_provided, \
                                      condition_exact_match, condition_subset, ignore_exact or self.dvfs_aware, min_entries, max_entries, \
                                      adjust_smem_to_gmem_ratio, adjust_gmem_to_smem_ratio, adjust_gmem_to_smem_select, \
                                      target_gpu_config=target_gpu_config, target_dvfs_supply_voltage=target_dvfs_supply_voltage, target_dvfs_idle_power=target_dvfs_idle_power, \
                                      verbose=verbose, return_power_flag=return_power_flag, predict_with_smaller_kernels=predict_with_smaller_kernels, use_precomputed_coeffs=use_precomputed_coeffs)

        else:
            raise NotImplementedError()
            

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
    
            
