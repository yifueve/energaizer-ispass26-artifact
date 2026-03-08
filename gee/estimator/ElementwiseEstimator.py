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
from scipy.optimize import minimize

import sys
sys.path.append('..')

from gee.kernel_parser import ElementwiseParser
from gee.analytical_model import ElementwiseAnalyticalModel
from gee.optimization_utils import optimize

from gee.estimator.BaseEstimator import BaseEstimator

class ElementwiseEstimator(BaseEstimator):
    def __init__(self, lut_config=None, gpu_config=None, \
                 dvfs_aware=False, dvfs_inference_mode='single_source', dvfs_supply_voltage={}, dvfs_idle_power={}, \
                 lut_folder_abs_path='/mnt/c/Users/KyungmiLee/Documents/gpu-energy-estimation/lut', \
                 multiple_configs=False, gpu_configs={}, dvfs_idle_power_configs={}, dvfs_supply_voltage_configs={}):
        
        op = 'elementwise'
        ops_supported = ['pointwise_mul', 'pointwise_add', \
                         'scalar_mul', 'scalar_add', \
                         'typecast_to_fp32', 'typecast_to_bf16', \
                         'relu', 'gelu', 'silu', 'tanh', 'sigmoid', \
                         'unspecified_activation', 'unspecified_tensor', 'unspecified_scalar']
        
        super().__init__(op, ops_supported, gpu_config=gpu_config, \
                         dvfs_aware=dvfs_aware, dvfs_inference_mode=dvfs_inference_mode, \
                         dvfs_supply_voltage=dvfs_supply_voltage, dvfs_idle_power=dvfs_idle_power, \
                         multiple_configs=multiple_configs, gpu_configs=gpu_configs, dvfs_idle_power_configs=dvfs_idle_power_configs, dvfs_supply_voltage_configs=dvfs_supply_voltage_configs)
        
        self.kernel_parser = ElementwiseParser()

        if self.multiple_configs:
            self.analytical_models = {}
            for key in self.gpu_configs.keys():
                self.analytical_models[key] = ElementwiseAnalyticalModel(self.gpu_configs[key], \
                                                                         dvfs_supply_voltage=self.dvfs_supply_voltage_configs[key], \
                                                                         dvfs_idle_power=self.dvfs_idle_power_configs[key])
            self.analytical_model = None
        else:
            self.analytical_model = ElementwiseAnalyticalModel(self.gpu_config, \
                                                               dvfs_supply_voltage=self.dvfs_supply_voltage, dvfs_idle_power=self.dvfs_idle_power)
            self.analytical_models = None

        # From lut_config, obtain database 
        # -> instead of keeping separate databases for different op types,
        #    we will keep them in one database (misc. and they are very similar anyway)
        self.model_database = {}
        self.model_database_main_id = {}
        self.kernel_database = {}
        self.model_database_path = {}

        # Multiple config case
        self.model_database_config_key = {}

        # Precomputed coeffs
        self.model_coeff_precomputed = {}

        for key in lut_config.keys():
            if key != op:
                continue
            self.model_database['elementwise'] = []
            self.kernel_database['elementwise'] = []
            self.model_database_config_key['elementwise'] = []
            self.model_coeff_precomputed['elementwise'] = []
            self.model_database_path['elementwise'] = []

            for i, elem in enumerate(lut_config[key]):
                df = pd.read_csv(os.path.join(lut_folder_abs_path, elem['path']))
                if elem['use_for_model']:
                    self.model_database['elementwise'].append(df)
                    if self.multiple_configs:
                        self.model_database_config_key['elementwise'].append(elem['gpu_config_key'])
                if elem['use_for_kernel']:
                    self.kernel_database['elementwise'].append(df)
                if elem['main']:
                    self.model_database_main_id['elementwise'] = i
                self.model_database_path['elementwise'].append(os.path.join(lut_folder_abs_path, elem['path']))
                    
                if ('precomputed_coeff' in elem.keys()) and elem['precomputed_coeff']:
                        coeffs = pd.read_csv(elem['precomputed_coeff_path'])
                        self.model_coeff_precomputed['elementwise'].append(coeffs)
        
        for key, df_arr in self.kernel_database.items():
            self.kernel_database[key] = pd.concat(df_arr)

    def build(self, no_build_model=False):
        # Kernel predictor
        for key, df in self.kernel_database.items():
            self.kernel_parser.parse_dataframe(df)
        self.build_kernel_predictor()

        if no_build_model:
            return

        # Model database
        for key, df_arr in self.model_database.items():
            for i, df in enumerate(df_arr):
                self.kernel_parser.parse_dataframe(df)

                if self.multiple_configs:
                    config_key = self.model_database_config_key[key][i]
                    analytical_model = self.analytical_models[config_key]
                else:
                    analytical_model = self.analytical_model

                analytical_model.model(df, train=True)

    def _get_kernel_fields(self, op):
        return ['kernel_type', 'max_concurrent_block']

    def _build_kernel_map(self):
        self._kernel_map = {}
        for key, df in self.kernel_database.items():
            # if key[0] == 'layernorm':
            #     continue
            fields = self._get_kernel_fields(key[0])
            unique_combinations = df[fields].drop_duplicates()
            unique_combinations = unique_combinations.to_dict('records')
            km = {idx: value for idx, value in enumerate(unique_combinations)}
            self._kernel_map[key] = km
    
    def build_kernel_predictor(self):
        self.kernel_predictor = {}
        # self._build_kernel_map()

        # def get_idx(row, km, fields):
        #     tmp = {}
        #     for k in fields:
        #         tmp[k] = row[k]
        #     return list(km.keys())[list(km.values()).index(tmp)]

        for key, df in self.kernel_database.items():
            mapper = {}
            precs = list(df['prec'].unique())

            for op in self.ops_supported:
                for p in precs:
                    _df = df.loc[(df['op'] == op) & (df['prec'] == p)]
                    if len(_df) == 0:
                        continue
                    k = df['kernel_type'].mode()[0]
                    concurrency = _df['max_concurrent_block'].mode()[0]
                    mapper[(op, p)] = (k, concurrency)
            
            self.kernel_predictor[key] = mapper

            # km = self._kernel_map[key]
            # fields = self._get_kernel_fields(key[0])
            # df['kernel_id'] = df.apply(lambda row: get_idx(row, km, fields), axis=1)
            # predictor = tree.DecisionTreeClassifier()
            # predictor = predictor.fit(df[['op', 'prec', 'dim']], df['kernel_id'])
            # self.kernel_predictor[key] = predictor
    
    def predict_kernel(self, query):
        kernel_predictor = self.kernel_predictor['elementwise'] # this should be 'elementwise'

        kernel_predictor_keys = list(set([x[0] for x in kernel_predictor.keys()]))

        if query['op'] not in kernel_predictor_keys:
            if query['op'] == 'unspecified_activation':
                query['op'] = 'relu' # as default
            elif query['op'] == 'unspecified_tensor':
                query['op'] = 'pointwise_add'
            elif query['op'] == 'unspecified_scalar':
                query['op'] = 'scalar_add'
            else:
                print(query)
                raise NotImplementedError("Unrecognized opration type")
        
        predicted_kernel_type, concurrency = kernel_predictor[(query['op'], query['prec'])]
        query['kernel_type'] = predicted_kernel_type
        query['max_concurrent_block'] = concurrency

        # km = self._kernel_map[('elementwise',)]
        # predicted_kernel_id = kernel_predictor.predict([[query['op'], query['prec'], query['dim']]])[0]

        query = self.kernel_parser.calculate_kernel_info_froom_prediction(query)

    def save_fixed_coeff_map(self, save_to):
        for key, df_arr in self.model_database.items():
            for i, df in enumerate(df_arr):
                lutname = self.model_database_path[key][i].split('/')[-1]
                p = lutname[:-4] + '_coeffs.csv'
                p = os.path.join(save_to, p)

                print("Saving the coefficients to the path: {}".format(p))

                # get the coeff map
                coeffs = self._build_fixed_coeff_map(df)

                # save it
                coeffs = pd.DataFrame(coeffs)
                coeffs.to_csv(p, index=False)

    def _build_fixed_coeff_map(self, df):
        coeffs = []
        precs = list(df['prec'].unique())

        for op in self.ops_supported:    
            for p in precs:
                coeff_dict = {}
                ref = df.loc[(df['op'] == op) & (df['prec'] == p)]
                coeff_dict['op'] = op
                coeff_dict['prec'] = p

                # perf
                try:
                    z = ref[['t_estimated']].to_numpy()
                    y = ref['cycles'].values
                    lambdas = optimize(z, y) 
                    perf_success = True
                except:
                    perf_success = False

                # power
                try:
                    self._power_activity_factors(ref, lambdas, True, False)
                    z = ref[['a_dram', 'a_l2', 'a_fp', 'a_other']].to_numpy()
                    y = ref['power'].to_numpy()
                    p_dram, p_l2, p_fp, p_other, p_static = optimize(z, y)
                    power_success = True
                except:
                    power_success = False

                if (perf_success and power_success):
                    coeff_dict['perf_coeffs'] = (lambdas[0], lambdas[1])
                    coeff_dict['power_coeffs'] = (p_dram, p_l2, p_fp, p_other, p_static)
                else:
                    coeff_dict['perf_coeffs'] = -1
                    coeff_dict['power_coeffs'] = -1

                coeffs.append(coeff_dict)
        
        return coeffs


    def get_references(self, query, max_entries=20, ignore_exact=False):
        prec = query['prec']

        lut = self.model_database['elementwise'][self.model_database_main_id['elementwise']]

        lut = lut.loc[(lut['op'] == query['op']) & (lut['prec'] == query['prec'])]

        exact_match_found = False
        exact = lut.loc[lut['dim'] == query['dim']]
        if (len(exact) > 0) and (not ignore_exact):
            flag = 'exact'
            ref = exact
            exact_match_found = True

        if not exact_match_found:
            ref = lut
            if len(ref) > max_entries:
                ref['abs_wave_diff'] = np.abs(ref['n_waves'] - query['n_waves'])

                ref.sort_values(by='abs_wave_diff', ascending=True, inplace=True)
                ref.reset_index(drop=True, inplace=True)
                ref = ref[:max_entries]
            flag = 'ref'
        
        return flag, ref
    
    def _predict_time(self, query, ref, fusion=False, \
                      use_precomputed_coeffs=False, precomputed_coeffs=None, verbose=False):
        try:
            if not use_precomputed_coeffs:
                z = ref[['t_estimated']].to_numpy()
                y = ref['cycles'].values

                lambdas = optimize(z, y)
            else:
                lambdas = precomputed_coeffs[0]
                if (type(lambdas) == str):
                    lambdas = eval(lambdas[1:-1])

            constant_term = lambdas[1]
            if fusion:
                constant_term *= 1 / query['fused_op_count']

            estimated_time = lambdas[0] * query['t_estimated'] + constant_term
            estimated_time /= (query['avg_freq'] * 10 ** 3)

            if verbose:
                print("Time estimation coefficients ---")
                print("First-order: {:.4f} | Other (const): {:.4f}".format(lambdas[0], lambdas[1]))
        
        except Exception as e:
            if verbose:
                print(e)

            z = ref[['dim']].to_numpy()
            y = ref['cycles'].values

            lambdas = optimize(z, y)
            estimated_time = lambdas[0] * query['dim'] + lambdas[1]
            estimated_time /= (query['avg_freq'] * 10 ** 3)

            lambdas = None
        
        return estimated_time, lambdas
    
    def _power_activity_factors(self, df, lambdas, train=True, \
                                dvfs=False, fusion=False):
        
        l_estimate, l_other = lambdas
        if fusion and not train:
            l_other *= 1 / df['fused_op_count']

        df['time_estimated_corrected'] = l_estimate * df['t_estimated'] +  l_other

        df['weight_dram'] = df['t_estimated'] * l_estimate / df['time_estimated_corrected']
        df['weight_l2'] = df['t_estimated'] * l_estimate / df['time_estimated_corrected']
        df['weight_fp'] = df['t_estimated'] * l_estimate / df['time_estimated_corrected']

        df['a_dram'] = df['a_dram_precorrection'] * df['weight_dram']
        df['a_l2'] = df['a_l2_precorrection'] * df['weight_l2']
        df['a_fp'] = df['a_fp_precorrection'] * df['weight_fp']
        df['a_other'] = l_other / df['time_estimated_corrected']

        if train:
            df['power'] = (df['energy'] / df['time']) * 1000.

        if dvfs:
            # df['gpu_supply_voltage'] = df['avg_freq'].apply(lambda x: dvfs_supply_voltage[str(int((x - 210) / 15) * 15 + 210)]) # mV
            # max_voltage = self.dvfs_max_core_voltage
            # df['gpu_supply_voltage_norm'] = df['gpu_supply_voltage'].apply(lambda x: x / max_voltage)
            # max_freq = self.dvfs_max_core_freq
            # df['gpu_freq_norm'] = df['avg_freq'].apply(lambda x: x / max_freq)
            
            # df['dvfs_scale_factor'] = df['gpu_supply_voltage_norm'] ** 2 * df['gpu_freq_norm'] # V^2 * f
            df['a_l2'] = df['a_l2'] * df['dvfs_scale_factor']
            df['a_fp'] = df['a_fp'] * df['dvfs_scale_factor']
            df['a_other'] = df['a_other'] * df['dvfs_scale_factor']

            df['extrapolation_sm_scale_factor'] = df['extrapolation_sm_scale_factor'].apply(lambda x: x if x > 0 else 1)
            df['a_fp'] = df['a_fp'] * df['extrapolation_sm_scale_factor']

            # df['hbm_freq_norm'] = df['hbm_freq'] / self.dvfs_max_hbm_freq
            df['a_dram'] = df['a_dram'] * df['hbm_freq_norm']

            if train: 
                # df['idle_power'] = df['avg_freq'].apply(lambda x: dvfs_idle_power[str(int((x - 210) / 15) * 15 + 210)]) # W
                df['dynamic_power'] = df['power'] - df['idle_power']

    def _predict_power(self, query, ref, lambdas, \
                       dvfs=False, fusion=False, use_precomputed_coeffs=False, precomputed_coeffs=None, verbose=False):
        
        try:
            query = pd.DataFrame([query], index=[0])
            self._power_activity_factors(query, lambdas, False, dvfs, fusion)
            query = dict(query.loc[0])

            if not use_precomputed_coeffs:
                self._power_activity_factors(ref, lambdas, True, dvfs)

                z = ref[['a_dram', 'a_l2', 'a_fp', 'a_other']].to_numpy()

                if not dvfs:
                    y = ref['power'].to_numpy()
                    p_dram, p_l2, p_fp, p_other, p_static = optimize(z, y)
                else:
                    y = ref['dynamic_power'].to_numpy()
                    p_dram, p_l2, p_fp, p_other = optimize(z, y, const=False)
            else:
                assert(dvfs == False)
                coeffs = precomputed_coeffs[1]
                if (type(coeffs) == str):
                    coeffs = eval(coeffs[1:-1])
                (p_dram, p_l2, p_fp, p_other, p_static) = coeffs

            if not dvfs:
                estimated_power = p_dram * query['a_dram'] + \
                                  p_l2 * query['a_l2'] + \
                                  p_fp * query['a_fp'] + \
                                  p_other * query['a_other'] + \
                                  p_static
            else:
                estimated_power = p_dram * query['a_dram'] + \
                                  p_l2 * query['a_l2'] + \
                                  p_fp * query['a_fp'] + \
                                  p_other * query['a_other']
                estimated_power += query['idle_power']

            if verbose:
                print("Power estimation coeffcients ---")
                if not dvfs:
                    print("DRAM: {:.4f} | L2: {:.4f} | Fp: {:.4f} | Other: {:.4f} | Static: {:.4f}".format(p_dram, p_l2, p_fp, p_other, p_static))
                else:
                    print("DRAM: {:.4f} | L2: {:.4f} | Fp: {:.4f} Other: {:.4f}".format(p_dram, p_l2, p_fp, p_other))

                print(ref.to_markdown())

        except Exception as e:
            if verbose:
                print(e)

            z = ref[['dim']].to_numpy()
            y = (ref['energy'] / ref['time'] * 1000.).to_numpy()

            lambdas = optimize(z, y)
            estimated_power = lambdas[0] * query['dim'] + lambdas[1]

        return estimated_power
    
    def _get_precomputed_coeffs(self, query):
        coeffs = self.model_coeff_precomputed['elementwise'][self.model_database_main_id['elementwise']]
        ref = coeffs.loc[(coeffs['op'] == query['op']) & (coeffs['prec'] == query['prec'])]

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
    
    def predict(self, query, op, target_freq, max_entries=20, \
                target_gpu_config=None, target_dvfs_supply_voltage=None, target_dvfs_idle_power=None, \
                 use_precomputed_coeffs=False, \
                verbose=False, **kwargs):
        
        self.predict_kernel(query)
        if verbose:
            print("Predicted kernel type for this elementwise op: ", query['kernel_type'])
        
        if target_freq is None:
            query['avg_freq'] = 1000 # 1GHz
        else:
            query['avg_freq'] = target_freq

        if ('fusion' in query.keys()) and (query['fusion'] == True):
            adjust_gmem_to_sm_ratio = 0 if query['fusion_ignore_in'] else 1
            adjust_sm_to_gmem_ratio = 0 if query['fusion_ignore_out'] else 1
            fusion = True
        else:
            adjust_gmem_to_sm_ratio = 1
            adjust_sm_to_gmem_ratio = 1
            fusion = False

        query = pd.DataFrame([query], index=[0])
        if target_gpu_config is not None:
            target_gpu_analytical_model = ElementwiseAnalyticalModel(target_gpu_config, \
                                                                     dvfs_supply_voltage=target_dvfs_supply_voltage, \
                                                                     dvfs_idle_power=target_dvfs_idle_power)
            target_gpu_analytical_model.model(query, False, adjust_gmem_to_sm_ratio=adjust_gmem_to_sm_ratio, adjust_sm_to_gmem_ratio=adjust_sm_to_gmem_ratio)
            del target_gpu_analytical_model
        else:
            if self.multiple_configs:
                print("Error: multiple GPU configuration in the database is only supported for extrapolation cases!")
                exit()
            self.analytical_model.model(query, False, adjust_gmem_to_sm_ratio=adjust_gmem_to_sm_ratio, adjust_sm_to_gmem_ratio=adjust_sm_to_gmem_ratio)
        query = dict(query.loc[0])

        precompute_success = False
        precomputed_coeffs = None
        if use_precomputed_coeffs:
            flag = 'precompute'
            precomputed_coeffs = self._get_precomputed_coeffs(query)
            ref = None

            if ((precomputed_coeffs[0]) == -1):
                precompute_success = False
            else:
                precompute_success = True

        if (not use_precomputed_coeffs) or (not precompute_success):
            flag, ref = self.get_references(query, max_entries=max_entries, ignore_exact=(self.dvfs_aware or fusion))

            # Empty target frequency -> only for non-multiple config cases
            # TODO: Add warning
            if (target_freq is None) and (flag == 'ref'):
                target_freq = ref['avg_freq'].mean()
                query['avg_freq'] = target_freq
                query = pd.DataFrame([query], index=[0])
                self.analytical_model.model(query, train=False, adjust_gmem_to_sm_ratio=adjust_gmem_to_sm_ratio, adjust_sm_to_gmem_ratio=adjust_sm_to_gmem_ratio)
                query = dict(query.loc[0])

            if verbose:
                print("Reference status: {} | Reference entry size: {}".format(flag, len(ref)))
                # print(ref.to_markdown())
                
            # DVFS Inference Mode is all, then use all sources
            if self.dvfs_inference_mode == 'all':
                dfs_ref = []
                prec = query['prec']
                for dvfs_df in self.model_database['elementwise']:
                    _df = pd.merge(dvfs_df, ref, on=['op', 'prec', 'dim'], how='inner', indicator=True, suffixes=[None, 'ref'])
                    _df = _df.loc[:, dvfs_df.columns]
                    dfs_ref.append(_df)
                ref = pd.concat(dfs_ref, ignore_index=True)

                # if verbose:
                #     print(ref.to_markdown())

        if flag == 'exact':
            return (ref.iloc[0]['time'], -1, ref.iloc[0]['energy'])
        else:
            estimated_time, coeffs = self._predict_time(query, ref, fusion=fusion, \
                                                        use_precomputed_coeffs=(use_precomputed_coeffs and precompute_success), \
                                                        precomputed_coeffs=precomputed_coeffs, verbose=verbose)
            estimated_power = self._predict_power(query, ref, coeffs, \
                                                  dvfs=self.dvfs_aware, fusion=fusion, \
                                                  use_precomputed_coeffs=(use_precomputed_coeffs and precompute_success), \
                                                  precomputed_coeffs=precomputed_coeffs,verbose=verbose)
            if type(estimated_power) == pd.core.series.Series:
                estimated_power = estimated_power.values[0]

            power_cap = self.gpu_config['power_cap'] if target_gpu_config is None else target_gpu_config['power_cap']
            power = min(estimated_power, power_cap)
            estimated_energy = power * estimated_time / 1000.
            return (estimated_time, estimated_power, estimated_energy)





