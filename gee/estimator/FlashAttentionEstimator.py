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

from gee.kernel_parser import FlashAttentionParser
from gee.analytical_model import FlashAttentionAnalyticalModel
from gee.optimization_utils import optimize

from gee.estimator.BaseEstimator import BaseEstimator

class FlashAttentionEstimator(BaseEstimator):
    def __init__(self, lut_config=None, gpu_config=None, \
                 dvfs_aware=False, dvfs_inference_mode='single_source', dvfs_supply_voltage=None, dvfs_idle_power=None, \
                 lut_folder_abs_path='/mnt/c/Users/KyungmiLee/Documents/gpu-energy-estimation/lut', \
                 multiple_configs=False, gpu_configs={}, dvfs_idle_power_configs={}, dvfs_supply_voltage_configs={}):
        
        op = 'flashattention'
        ops_supported = ['flashattention_v2']
        super().__init__(op, ops_supported, gpu_config=gpu_config, \
                         dvfs_aware=dvfs_aware, dvfs_inference_mode=dvfs_inference_mode, dvfs_supply_voltage=dvfs_supply_voltage, dvfs_idle_power=dvfs_idle_power, \
                         multiple_configs=multiple_configs, gpu_configs=gpu_configs, dvfs_idle_power_configs=dvfs_idle_power_configs, dvfs_supply_voltage_configs=dvfs_supply_voltage_configs)

        self.kernel_parser = FlashAttentionParser()

        if self.multiple_configs:
            self.analytical_models = {}
            for key in self.gpu_configs.keys():
                self.analytical_models[key] = FlashAttentionAnalyticalModel(self.gpu_configs[key], \
                                                                            dvfs_supply_voltage=self.dvfs_supply_voltage_configs[key], \
                                                                            dvfs_idle_power=self.dvfs_idle_power_configs[key])
                self.analytical_model = None
        
        else:
            self.analytical_model = FlashAttentionAnalyticalModel(self.gpu_config, \
                                                                  dvfs_supply_voltage=self.dvfs_supply_voltage, \
                                                                  dvfs_idle_power=self.dvfs_idle_power)
            self.analytical_models = None

        self.model_database = {}
        self.model_database_main_id = {}
        self.kernel_database = {}

        self.model_database_config_key = {} # for multi configuration cases

        for key in lut_config.keys():
            if key not in ops_supported:
                continue
            for prec in lut_config[key].keys():
                self.model_database[(key, prec)] = []
                self.kernel_database[(key, prec)] = []

                if self.multiple_configs:
                    self.model_database_config_key[(key, prec)] = []

                for i, elem in enumerate(lut_config[key][prec]):
                    df = pd.read_csv(os.path.join(lut_folder_abs_path, elem['path']))
                    if elem['use_for_model']:
                        self.model_database[(key, prec)].append(df)
                        if self.multiple_configs:
                            self.model_database_config_key[(key, prec)].append(elem['gpu_config_key'])
                    if elem['use_for_kernel']:
                        self.kernel_database[(key, prec)].append(df)
                    if elem['main']:
                        self.model_database_main_id[(key, prec)] = i
        
        for key, df_arr in self.kernel_database.items():
            self.kernel_database[key] = pd.concat(df_arr)
        
    def build(self):
        for key, df in self.kernel_database.items():
            self.kernel_parser.parse_dataframe(df, key[0])
        
        self.build_kernel_predictor()

        for key, df_arr in self.model_database.items():
            for i, df in enumerate(df_arr):
                self.kernel_parser.parse_dataframe(df, key[0])

                if self.multiple_configs:
                    config_key = self.model_database_config_key[key][i]
                    analytical_model = self.analytical_models[config_key]
                else:
                    analytical_model = self.analytical_model
                
                analytical_model.model(df, train=True)

    def _get_kernel_fields(self, op):
        if op == 'flashattention_v2':
            return ['block_r', 'block_c', 'head_dim', 'n_warps_per_block', 'max_concurrent_block']
        else:
            raise NotImplementedError()

    def _build_kernel_map(self):
        self._kernel_map = {}
        for key, df in self.kernel_database.items():
            fields = self._get_kernel_fields(key[0])
            unique_combinations = df[fields].drop_duplicates()
            unique_combinations = unique_combinations.to_dict('records')
            km = {idx: value for idx, value in enumerate(unique_combinations)}
            self._kernel_map[key] = km

    def build_kernel_predictor(self):
        self.kernel_predictor = {}
        self._build_kernel_map()

        def get_idx(row, km, fields):
            tmp = {}
            for k in fields:
                tmp[k] = row[k]
            return list(km.keys())[list(km.values()).index(tmp)]
        
        for key, df in self.kernel_database.items():
            km = self._kernel_map[key]
            fields = self._get_kernel_fields(key[0])
            df['kernel_id'] = df.apply(lambda row: get_idx(row, km, fields), axis=1)
            predictor = tree.DecisionTreeClassifier()
            predictor = predictor.fit(df[['batch', 'n_head', 'seq_len', 'head_dim']], df['kernel_id'])
            self.kernel_predictor[key] = predictor

    def predict_kernel(self, query, op):
        assert(op in self.ops_supported)
        prec = query['prec']

        kernel_predictor = self.kernel_predictor[(op, prec)]

        km = self._kernel_map[(op, prec)]
        predicted_kernel_id = kernel_predictor.predict([[query['batch'], query['n_head'], query['seq_len'], query['head_dim']]])[0]
        kernel_info = km[predicted_kernel_id]

        for k, v in kernel_info.items():
            query[k] = v

        query = self.kernel_parser.calculate_kernel_info_from_prediction(query, kernel_info)

    def get_references(self, query, op, max_entries=20, ignore_exact=False):
        assert (op in self.ops_supported)
        prec = query['prec']

        lut = self.model_database[(op, prec)][self.model_database_main_id[(op, prec)]]

        exact_match_found = False
        exact = lut.loc[(lut['batch'] == query['batch']) & \
                        (lut['n_head'] == query['n_head']) & \
                        (lut['seq_len'] == query['seq_len']) & \
                        (lut['head_dim'] == query['head_dim'])]
        if (len(exact) > 0) and (not ignore_exact):
            flag = 'exact'
            ref = exact
            exact_match_found = True

        if not exact_match_found:
            ref = lut.loc[(lut['block_r'] == query['block_r']) & \
                          (lut['block_c'] == query['block_c']) & \
                          (lut['n_warps_per_block'] == query['n_warps_per_block'])]
            
            if len(ref) > max_entries:
                ref['abs_wave_diff'] = np.abs(ref['n_waves'] - query['n_waves'])
                ref['abs_dim_diff'] = np.abs(ref['head_dim'] - query['head_dim'])

                if query['n_waves'] <= query['max_concurrent_block']:
                    ref.sort_values(by=['abs_wave_diff', 'abs_dim_diff'], ascending=[True, True], inplace=True)
                else:
                    ref.sort_values(by=['abs_dim_diff', 'abs_wave_diff'], ascending=[True, True], inplace=True)

                ref.reset_index(drop=True, inplace=True)
                ref = ref[:max_entries]

            flag = 'ref'
        
        return flag, ref
    
    def _predict_time(self, query, ref, target_freq, verbose=False):
        try:
            z = ref[['t_start', 't_work', 't_end']].to_numpy()
            y = ref['cycles'].to_numpy()

            lambda_1, lambda_2, lambda_3, lambda_4 = optimize(z, y)
            cycles_estimate = query['t_start'] * lambda_1 + query['t_work'] * lambda_2 + query['t_end'] * lambda_3 + lambda_4
            coeffs = (lambda_1, lambda_2, lambda_3, lambda_4)

            if verbose:
                print("Architecture aware modeling success!")
                print("Time estimate coefficients ---")
                print("Lambda1 (start): {:.4f} | Lambda2 (work): {:.4f} | Lambda3 (end): {:.4f} | Lambda4 (const): {:.4f}".format(lambda_1, lambda_2, lambda_3, lambda_4))

        except:
            # fall back: simple linear regression based on number of waves and head_dim
            z = ref['head_dim'].to_numpy()
            y = (ref['cycles'] / ref['n_waves']).to_numpy()
            slope, intercept = optimize(z, y)
            cycles_estimate = (query['head_dim'] * slope + intercept) * query['n_waves']
            coeffs = None

            if verbose:
                print("Architectural aware modeling failed. Fall back to linear regression")

        estimated_time = cycles_estimate / (target_freq * 10 ** 3)
        return estimated_time, coeffs
    
    def _power_activity_factors(self, df, lambdas, train=True, dvfs=False):
        lambda_1, lambda_2, lambda_3, lambda_4 = lambdas

        df['time_estimated_corrected'] = lambda_1 * df['t_start'] + lambda_2 * df['t_work'] + lambda_3 * df['t_end'] + lambda_4
        df['weight_start_full_capacity'] = lambda_1 * df['t_prologue_full_capacity'] * (df['n_waves'] - 1) / df['time_estimated_corrected']
        df['weight_start_last_wave'] = lambda_1 * df['t_prologue_last_wave'] / df['time_estimated_corrected']
        df['weight_work_full_capacity'] = lambda_2 * df['t_stage_full_capacity'] * (df['n_waves'] - 1) * df['iteration_stages'] / df['time_estimated_corrected']
        df['weight_work_last_wave'] = lambda_2 * df['t_stage_last_wave'] * df['iteration_stages'] / df['time_estimated_corrected']
        df['weight_end_full_capacity'] = lambda_3 * df['t_epilogue_full_capacity'] / df['time_estimated_corrected']
        df['weight_end_last_wave'] = lambda_3 * df['t_epilogue_last_wave'] / df['time_estimated_corrected']

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
        df['a_tc'] = df['weight_start_full_capacity'] * df['a_tc_start_full_capacity'] + \
                    df['weight_start_last_wave'] * df['a_tc_start_last_wave'] + \
                    df['weight_work_full_capacity'] * df['a_tc_work_full_capacity'] + \
                    df['weight_work_last_wave'] * df['a_tc_work_last_wave'] + \
                    df['weight_end_full_capacity'] * df['a_tc_end_full_capacity'] + \
                    df['weight_end_last_wave'] * df['a_tc_end_last_wave']
        df['a_fp'] = df['weight_start_full_capacity'] * df['a_fp_start_full_capacity'] + \
                    df['weight_start_last_wave'] * df['a_fp_start_last_wave'] + \
                    df['weight_work_full_capacity'] * df['a_fp_work_full_capacity'] + \
                    df['weight_work_last_wave'] * df['a_fp_work_last_wave'] + \
                    df['weight_end_full_capacity'] * df['a_fp_end_full_capacity'] + \
                    df['weight_end_last_wave'] * df['a_fp_end_last_wave']
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
            df['a_tc'] = df['a_tc'] * df['dvfs_scale_factor']
            df['a_fp'] = df['a_fp'] * df['dvfs_scale_factor']
            df['a_other'] = df['a_other'] * df['dvfs_scale_factor']

            df['extrapolation_sm_scale_factor'] = df['extrapolation_sm_scale_factor'].apply(lambda x: x if x > 0 else 1)
            df['extrapolation_tc_scale_factor'] = df['extrapolation_tc_scale_factor'].apply(lambda x: x if x > 0 else 1)
            df['a_smem'] = df['a_smem'] * df['extrapolation_sm_scale_factor']
            df['a_tc'] = df['a_tc'] * df['extrapolation_sm_scale_factor'] * df['extrapolation_tc_scale_factor']
            df['a_fp'] = df['a_fp'] * df['extrapolation_sm_scale_factor']

            # df['hbm_freq_norm'] = df['hbm_freq'] / self.dvfs_max_hbm_freq
            df['a_dram'] = df['a_dram'] * df['hbm_freq_norm']

            if train:
                # df['power'] = (df['energy'] / df['time']) * 1000. 
                # df['idle_power'] = df['avg_freq'].apply(lambda x: dvfs_idle_power[str(int((x - 210) / 15) * 15 + 210)]) # W
                df['dynamic_power'] = df['power'] - df['idle_power']

    def _predict_power(self, query, ref, lambdas, dvfs=False, verbose=False):
        try:
            query = pd.DataFrame([query], index=[0])
            self._power_activity_factors(query, lambdas, False, dvfs)
            query = dict(query.loc[0])
            self._power_activity_factors(ref, lambdas, True, dvfs)

            z = ref[['a_dram', 'a_l2', 'a_smem', 'a_tc', 'a_fp', 'a_other']].to_numpy()

            if not dvfs:
                y = (1000 * ref['energy'].values / ref['time'].values)
                p_dram, p_l2, p_smem, p_tc, p_fp, p_other, p_static = optimize(z, y)
            else:
                y = ref['dynamic_power'].to_numpy()
                p_dram, p_l2, p_smem, p_tc, p_fp, p_other = optimize(z, y, const=False)
            
            if not dvfs:
                power_estimate = p_dram * query['a_dram'] + p_l2 * query['a_l2'] + p_smem * query['a_smem'] + p_tc * query['a_tc'] + p_fp * query['a_fp'] + p_other * query['a_other'] + p_static
            else:
                power_estimate = p_dram * query['a_dram'] + p_l2 * query['a_l2'] + p_smem * query['a_smem'] + p_tc * query['a_tc'] + p_fp * query['a_fp'] + p_other * query['a_other'] 
                power_estimate += query['idle_power']

            if verbose:
                print("Activity factor based power estimation success!")
                if not dvfs:
                    print("Not DVFS! Coefficients ---")
                    print("DRAM: {:.4f} | L2: {:.4f} | Smem: {:.4f} | TC: {:.4f} | FP: {:.4f} | Other: {:.4f} | Static: {:.4f}".format(p_dram, p_l2, p_smem, p_tc, p_fp, p_other, p_static))
                else:
                    print("DVFS Aware! Coefficients ---")
                    print("DRAM: {:.4f} | L2: {:.4f} | Smem: {:.4f} | TC: {:.4f} | FP: {:.4f} | Other: {:.4f}".format(p_dram, p_l2, p_smem, p_tc, p_fp, p_other))
                
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

        return power_estimate

    def predict(self, query, op, target_freq, max_entries=20, ignore_exact=False, \
                target_gpu_config=None, target_dvfs_supply_voltage=None, target_dvfs_idle_power=None, \
                verbose=False, **kwargs):
        if (op == 'flashattention_v2'):
            assert(set(['batch', 'n_head', 'seq_len', 'head_dim', 'prec'])).issubset(list(query.keys()))
        else:
            raise NotImplementedError()
        
        if 'precM' not in query.keys():
            query['precM'] = query['prec']
        if 'precA' not in query.keys():
            query['precA'] = query['prec']
        if 'useTensorCore' not in query.keys():
            query['useTensorCore'] = True
        
        self.predict_kernel(query, op)

        if verbose:
            print("Predicted kernel information:")
            for k, v in query.items():
                print(k, v)
        
        if target_freq is None:
            query['avg_freq'] = 900
        else:
            query['avg_freq'] = target_freq

        query = pd.DataFrame([query], index=[0])
        if target_gpu_config is not None:
            target_gpu_analytical_model = FlashAttentionAnalyticalModel(target_gpu_config, \
                                                                        dvfs_supply_voltage=target_dvfs_supply_voltage, \
                                                                        dvfs_idle_power=target_dvfs_idle_power)
            target_gpu_analytical_model.model(query, train=False)
            del target_gpu_analytical_model
        else:
            assert(self.multiple_configs == False)
            self.analytical_model.model(query, train=False)
        query = dict(query.loc[0])

        flag, ref = self.get_references(query, op, max_entries, ignore_exact=(self.dvfs_aware or ignore_exact))

        if verbose:
            print("Reference status: {} | Reference entry size {}".format(flag, len(ref)))

        # DVFS Inference Mode is all, then use all sources
        if self.dvfs_inference_mode == 'all':
            dfs_ref = []
            prec = query['prec']
            for dvfs_df in self.model_database[(op, prec)]:
                _df = pd.merge(dvfs_df, ref, on=['batch', 'n_head', 'seq_len', 'head_dim'], how='inner', indicator=True, suffixes=[None, 'ref'])
                _df = _df.loc[:, dvfs_df.columns]
                dfs_ref.append(_df)
            ref = pd.concat(dfs_ref, ignore_index=True)

            # if verbose:
            #     print(ref.to_markdown())

        if flag == 'exact':
            return (ref.iloc[0]['time'], -1, ref.iloc[0]['energy'])
        else:
            estimated_time, coeffs = self._predict_time(query, ref, target_freq=query['avg_freq'], verbose=verbose)
            estimated_power = self._predict_power(query, ref, coeffs, \
                                                  dvfs=self.dvfs_aware, verbose=verbose)
            if type(estimated_power) == pd.core.series.Series:
                estimated_power = estimated_power.values[0]

            power_cap = self.gpu_config['power_cap'] if target_gpu_config is None else target_gpu_config['power_cap']
            power = min(estimated_power, power_cap)
            estimated_energy = power * estimated_time / 1000.
            return (estimated_time, estimated_power, estimated_energy)

        

