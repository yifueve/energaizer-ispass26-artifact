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

from gee.kernel_parser import NonlinearParser
from gee.analytical_model import NonlinearAnalyticalModel
from gee.optimization_utils import optimize

from gee.estimator.BaseEstimator import BaseEstimator

class NonlinearEstimator(BaseEstimator):
    def __init__(self, lut_config=None, gpu_config=None, \
                 dvfs_aware=False, dvfs_inference_mode='single_source', dvfs_supply_voltage={}, dvfs_idle_power={}, \
                 lut_folder_abs_path='/mnt/c/Users/KyungmiLee/Documents/gpu-energy-estimation/lut', \
                 multiple_configs=False, gpu_configs={}, dvfs_idle_power_configs={}, dvfs_supply_voltage_configs={}):
        
        op = 'nonlinear'
        ops_supported = ['softmax', 'layernorm', 'softmax_fusion', 'layernorm_fusion']
        super().__init__(op, ops_supported, gpu_config=gpu_config, \
                         dvfs_aware=dvfs_aware, dvfs_inference_mode=dvfs_inference_mode, dvfs_supply_voltage=dvfs_supply_voltage, dvfs_idle_power=dvfs_idle_power, \
                         multiple_configs=multiple_configs, gpu_configs=gpu_configs, dvfs_idle_power_configs=dvfs_idle_power_configs, dvfs_supply_voltage_configs=dvfs_supply_voltage_configs)

        self.kernel_parser = NonlinearParser()

        if self.multiple_configs:
            self.analytical_models = {}
            for key in self.gpu_configs.keys():
                self.analytical_models[key] = NonlinearAnalyticalModel(self.gpu_configs[key], \
                                                                       dvfs_supply_voltage=self.dvfs_supply_voltage_configs[key], \
                                                                       dvfs_idle_power=self.dvfs_idle_power_configs[key])
                self.analytical_model = None
        else:
            self.analytical_model = NonlinearAnalyticalModel(self.gpu_config, \
                                                             dvfs_supply_voltage=self.dvfs_supply_voltage, \
                                                             dvfs_idle_power=self.dvfs_idle_power)
            self.analytical_models = None

        # From lut_config, obtain database 
        self.model_database = {}
        self.model_database_main_id = {}
        self.kernel_database = {}
        self.model_database_path = {}

        # Multiple config case
        self.model_database_config_key = {}

        # Precomputed coeffs
        self.model_coeff_precomputed = {}

        for key in lut_config.keys():
            if key not in ops_supported:
                continue
            for prec in lut_config[key].keys():
                self.model_database[(key, prec)] = []
                self.kernel_database[(key, prec)] = []
                self.model_database_path[(key, prec)] = []

                if self.multiple_configs:
                    self.model_database_config_key[(key, prec)] = []

                self.model_coeff_precomputed[(key, prec)] = []

                for i, elem in enumerate(lut_config[key][prec]):
                    df = pd.read_csv(os.path.join(lut_folder_abs_path, elem['path']))
                    if elem['use_for_model']:
                        self.model_database[(key, prec)].append(df)
                        if self.multiple_configs:
                            self.model_database_config_key[(key, prec)].append(elem['gpu_config_key'])
                    self.model_database_path[(key, prec)].append(os.path.join(lut_folder_abs_path, elem['path']))
                    if elem['use_for_kernel']:
                        self.kernel_database[(key, prec)].append(df)
                    if elem['main']:
                        self.model_database_main_id[(key, prec)] = i

                    if ('precomputed_coeff' in elem.keys()) and elem['precomputed_coeff']:
                        coeffs = pd.read_csv(elem['precomputed_coeff_path'])
                        self.model_coeff_precomputed[(key, prec)].append(coeffs)
        
        for key, df_arr in self.kernel_database.items():
            self.kernel_database[key] = pd.concat(df_arr)

    def build(self, no_build_model=False):
        # Parse the database - kernel first
        for key, df in self.kernel_database.items():
            self.kernel_parser.parse_dataframe(df, key[0])
        
        self.build_kernel_predictor()

        if no_build_model:
            return

        # Parse for model database
        for key, df_arr in self.model_database.items():
            for i, df in enumerate(df_arr):
                self.kernel_parser.parse_dataframe(df, key[0])

                if self.multiple_configs:
                    config_key = self.model_database_config_key[key][i]
                    analytical_model = self.analytical_models[config_key]
                else:
                    analytical_model = self.analytical_model

                analytical_model.model(df, key[0], train=True)

    def _get_kernel_fields(self, op):
        if (op == 'softmax') or (op == 'softmax_fmha'):
            return ['kernel_type', 'n_warps_per_block', 'warp_tile_batch', 'max_concurrent_block']
        elif op == 'layernorm':
            return ['max_concurrent_block']
        else:
            raise NotImplementedError()

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
        self._build_kernel_map()

        def get_idx(row, km, fields):
            tmp = {}
            for k in fields:
                tmp[k] = row[k]
            return list(km.keys())[list(km.values()).index(tmp)]
        
        for key, df in self.kernel_database.items():
            # if key[0] == 'layernorm':
            #     continue
            km = self._kernel_map[key]
            fields = self._get_kernel_fields(key[0])
            df['kernel_id'] = df.apply(lambda row: get_idx(row, km, fields), axis=1)
            predictor = tree.DecisionTreeClassifier()
            predictor = predictor.fit(df[['batch', 'dim']], df['kernel_id'])
            self.kernel_predictor[key] = predictor

    def predict_kernel(self, query, op):
        assert(op in self.ops_supported)
        prec = query['prec']

        if op == 'softmax_fusion':
            op = 'softmax'
        elif op == 'layernorm_fusion':
            op = 'layernorm'

        kernel_predictor = self.kernel_predictor[(op, prec)]

        km = self._kernel_map[(op, prec)]
        predicted_kernel_id = kernel_predictor.predict([[query['batch'], query['dim']]])[0]
        kernel_info = km[predicted_kernel_id]

        for k, v in kernel_info.items():
            query[k] = v
        
        query = self.kernel_parser.calculate_kernel_info_from_prediction(query, op)

    def save_fixed_coeff_map(self, save_to):
        for key, df_arr in self.model_database.items():
            for i, df in enumerate(df_arr):
                lutname = self.model_database_path[key][i].split('/')[-1]
                p = lutname[:-4] + '_coeffs.csv'
                p = os.path.join(save_to, p)

                print("Saving the coefficients to the path: {}".format(p))

                # get the kernelmap
                kernelmap = self._kernel_map[key]
                kernelfields = self._get_kernel_fields(key[0])

                # get the coeff map
                coeffs = self._build_fixed_coeff_map(df, kernelmap, kernelfields)

                # save it
                coeffs = pd.DataFrame(coeffs)
                coeffs.to_csv(p, index=False)

    def _build_fixed_coeff_map(self, df, kernelmap, kernelfields):
        coeffs = []
        for _, kernel_field in kernelmap.items():
            coeff_dict = {}
            cond = True
            for f in kernelfields:
                cond = cond & (df[f] == kernel_field[f])
                coeff_dict[f] = kernel_field[f]
            ref = df.loc[cond]

            # perf
            try:
                z = ref[['t_estimated_gmem', 't_estimated_smem', 't_estimated_fp_inst', 't_estimated_xu_inst']].to_numpy()
                y = ref['cycles'].values
                lambdas = optimize(z, y)
                perf_success = True
            except:
                perf_success = False

            # power
            try:
                self._power_activity_factors(ref, lambdas, True, False)
                z = ref[['a_dram', 'a_l2', 'a_smem', 'a_fp', 'a_xu', 'a_other']].to_numpy()
                y = ref['power'].to_numpy()
                p_dram, p_l2, p_smem, p_fp, p_xu, p_other, p_static = optimize(z, y)
                power_success = True
            except:
                power_success = False

            if (perf_success and power_success):
                coeff_dict['perf_coeffs'] = tuple(lambdas)
                coeff_dict['power_coeffs'] = (p_dram, p_l2, p_smem, p_fp, p_xu, p_other, p_static)
            else:
                coeff_dict['perf_coeffs'] = -1
                coeff_dict['power_coeffs'] = -1

            coeffs.append(coeff_dict)
        
        return coeffs

    def get_references(self, query, op, max_entries=20, ignore_exact=False):
        assert(op in self.ops_supported)
        prec = query['prec']

        lut = self.model_database[(op, prec)][self.model_database_main_id[(op, prec)]]

        exact_match_found = False
        exact = lut.loc[(lut['batch'] == query['batch']) & (lut['dim'] == query['dim'])]
        if (len(exact) > 0) and (not ignore_exact):
            flag = 'exact'
            ref = exact
            exact_match_found = True

        if not exact_match_found:
            if op != 'layernorm':
                ref = lut.loc[lut['kernel_type'] == query['kernel_type']]
            else:
                ref = lut
            if len(ref) > max_entries:
                ref['abs_wave_diff'] = np.abs(ref['n_waves'] - query['n_waves'])

                if op == 'softmax':
                    ref['abs_temporal_diff'] = np.abs(ref['num_warp_tile_softmax_temporal'] - query['num_warp_tile_softmax_temporal'])
                elif op == 'layernorm':
                    ref['abs_temporal_diff'] = np.abs(ref['num_warp_tile_layernorm_temporal'] - query['num_warp_tile_layernorm_temporal'])

                # if query has small number of waves, prioritize waves 
                if query['n_waves'] <= query['max_concurrent_block']:
                    ref.sort_values(by=['abs_wave_diff', 'abs_temporal_diff'], ascending=[True, True], inplace=True)
                else:
                    ref.sort_values(by=['abs_temporal_diff', 'abs_wave_diff'], ascending=[True, True], inplace=True)
                
                ref.reset_index(drop=True, inplace=True)
                ref = ref[:max_entries]
            flag = 'ref'
        
        return flag, ref
    
    def _predict_time(self, query, ref, fusion=False, \
                      use_precomputed_coeffs=False, precomputed_coeffs=None, verbose=False):
        try:

            if not use_precomputed_coeffs:
                z = ref[['t_estimated_gmem', 't_estimated_smem', 't_estimated_fp_inst', 't_estimated_xu_inst']].to_numpy()
                y = ref['cycles'].values

                lambdas = optimize(z, y)
            else:
                lambdas = precomputed_coeffs[0]


            estimated_time = 0
            for i, k in enumerate(['t_estimated_gmem', 't_estimated_smem', 't_estimated_fp_inst', 't_estimated_xu_inst']):
                estimated_time += lambdas[i] * query[k]

            constant_term = lambdas[-1]
            if fusion:
                constant_term *= 1 / query['fused_op_count']
            estimated_time += constant_term
            estimated_time /= (query['avg_freq'] * 10** 3)

            if verbose:
                print("Time estimation coefficients ---")
                print("Gmem: {:.4f} | Smem: {:.4f} | Fp: {:.4f} | Xu: {:.4f} | Other: {:.4f}".format(lambdas[0], lambdas[1], lambdas[2], lambdas[3], lambdas[4]))
        except Exception as e:
            if verbose:
                print(e)
            # Estimate with a linear regression with the workload size
            ref['size'] = ref['batch'] * ref['dim']
            z = ref[['size']].to_numpy()
            y = ref['cycles'].values

            lambdas = optimize(z, y)
            estimated_time = lambdas[0] * query['batch'] * query['dim'] + lambdas[1]
            estimated_time /= (query['avg_freq'] * 10**3)
            
            lambdas = None

        return estimated_time, lambdas
    
    def _power_activity_factors(self, df, lambdas, train=True, \
                                dvfs=False, fusion=False):
        l_gmem, l_smem, l_fp, l_xu, l_other = lambdas

        if fusion and (not train):
            l_other *= 1 / df['fused_op_count']

        df['time_estimated_corrected'] = l_gmem * df['t_estimated_gmem'] + \
                                         l_smem * df['t_estimated_smem'] + \
                                         l_fp * df['t_estimated_fp_inst'] + \
                                         l_xu * df['t_estimated_xu_inst'] + \
                                         l_other
        
        df['weight_dram'] = df['t_estimated'] / df['time_estimated_corrected'] * l_gmem
        df['weight_l2'] = df['t_estimated'] / df['time_estimated_corrected'] * l_gmem
        df['weight_smem'] = df['t_estimated'] / df['time_estimated_corrected'] * l_smem
        df['weight_fp'] = df['t_estimated'] / df['time_estimated_corrected'] * l_fp
        df['weight_xu'] = df['t_estimated'] / df['time_estimated_corrected'] * l_xu
        
        df['a_dram'] = df['a_dram_precorrection'] * df['weight_dram']
        df['a_l2'] = df['a_l2_precorrection'] * df['weight_l2']
        df['a_smem'] = df['a_smem_precorrection'] * df['weight_smem']
        df['a_fp'] = df['a_fp_precorrection'] * df['weight_fp']
        df['a_xu'] = df['a_xu_precorrection'] * df['weight_xu']
        df['a_other'] = l_other / df['time_estimated_corrected']

        df['a_dram'] = df['a_dram'].apply(lambda x: max(x, 0))
        df['a_l2'] = df['a_l2'].apply(lambda x: max(x, 0))
        df['a_smem'] = df['a_smem'].apply(lambda x: max(x, 0))
        df['a_fp'] = df['a_fp'].apply(lambda x: max(x, 0))
        df['a_xu'] = df['a_xu'].apply(lambda x: max(x, 0))
        df['a_other'] = df['a_other'].apply(lambda x: max(x, 0))

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
            df['a_smem'] = df['a_smem'] * df['dvfs_scale_factor']
            df['a_fp'] = df['a_fp'] * df['dvfs_scale_factor'] 
            df['a_xu'] = df['a_xu'] * df['dvfs_scale_factor']
            df['a_other'] = df['a_other'] * df['dvfs_scale_factor']

            df['extrapolation_sm_scale_factor'] = df['extrapolation_sm_scale_factor'].apply(lambda x: x if x > 0 else 1)
            df['a_smem'] = df['a_smem'] * df['extrapolation_sm_scale_factor']
            df['a_fp'] = df['a_fp'] * df['extrapolation_sm_scale_factor'] 
            df['a_xu'] = df['a_xu'] * df['extrapolation_sm_scale_factor']

            # df['hbm_freq_norm'] = df['hbm_freq'] / self.dvfs_max_hbm_freq
            df['a_dram'] = df['a_dram'] * df['hbm_freq_norm']

            if train: 
                # df['idle_power'] = df['avg_freq'].apply(lambda x: dvfs_idle_power[str(int((x - 210) / 15) * 15 + 210)]) # W
                df['dynamic_power'] = df['power'] - df['idle_power']
    
    def _predict_power(self, query, ref, lambdas, \
                       dvfs=False, fusion=False, \
                       use_precomputed_coeffs=False, precomputed_coeffs=None, verbose=False):
        try:
            query = pd.DataFrame([query], index=[0])
            _lambdas = list(copy.deepcopy(lambdas))
            _lambdas = tuple(_lambdas)
            self._power_activity_factors(query, _lambdas, False, dvfs, fusion)
            query = dict(query.loc[0])

            if not use_precomputed_coeffs:
                self._power_activity_factors(ref, lambdas, True, dvfs)

                z = ref[['a_dram', 'a_l2', 'a_smem', 'a_fp', 'a_xu', 'a_other']].to_numpy()

                if not dvfs:
                    y = ref['power'].to_numpy()
                    p_dram, p_l2, p_smem, p_fp, p_xu, p_other, p_static = optimize(z, y)
                else:
                    y = ref['dynamic_power'].to_numpy()
                    p_dram, p_l2, p_smem, p_fp, p_xu, p_other = optimize(z, y, const=False)
            else:
                assert (dvfs == False)
                coeffs = precomputed_coeffs[1]
                (p_dram, p_l2, p_smem, p_fp, p_xu, p_other, p_static) = coeffs

            if not dvfs:
                estimated_power = p_dram * query['a_dram'] + \
                                  p_l2 * query['a_l2'] + \
                                  p_smem * query['a_smem'] + \
                                  p_fp * query['a_fp'] + \
                                  p_xu * query['a_xu'] + \
                                  p_other * query['a_other'] + \
                                  p_static
            else:
                estimated_power = p_dram * query['a_dram'] + \
                                  p_l2 * query['a_l2'] + \
                                  p_smem * query['a_smem'] + \
                                  p_fp * query['a_fp'] + \
                                  p_xu * query['a_xu'] + \
                                  p_other * query['a_other']
                estimated_power += query['idle_power']
            if verbose:
                print("Power estimation coeffcients ---")
                if not dvfs:
                    print("DRAM: {:.4f} | L2: {:.4f} | Smem: {:.4f} | Fp: {:.4f} | Xu: {:.4f} | Other: {:.4f} | Static: {:.4f}".format(p_dram, p_l2, p_smem, p_fp, p_xu, p_other, p_static))
                else:
                    print("DRAM: {:.4f} | L2: {:.4f} | Smem: {:.4f} | Fp: {:.4f} | Xu: {:.4f} | Other: {:.4f}".format(p_dram, p_l2, p_smem, p_fp, p_xu, p_other))
                # print(ref[['batch', 'dim', 'a_dram', 'a_l2', 'a_smem', 'a_fp', 'a_xu', 'a_other', 'power']].to_markdown())
                
                print("Corrected Lambdas: ", _lambdas)
                print("Original Lambdas ", lambdas)

        except Exception as e:
            if verbose:
                print(e)
                # print(ref[['batch', 'dim', 'a_dram', 'a_l2', 'a_smem', 'a_fp', 'a_xu', 'a_other', 'power']].to_markdown())
            
            # Linear regression similar as in latency
            ref['size'] = ref['batch'] * ref['dim']
            z = ref[['size']].to_numpy()
            y = (ref['energy'] / ref['time'] * 1000.).to_numpy()

            lambdas = optimize(z, y)
            estimated_power = lambdas[0] * query['batch'] * query['dim'] + lambdas[1]

        return estimated_power
        
    def _get_precomputed_coeffs(self, query, op):
        assert(op in self.ops_supported)
        prec = query['prec']
        coeffs = self.model_coeff_precomputed[(op, prec)][self.model_database_main_id[(op, prec)]]
        kernelfields = self._get_kernel_fields(op)
        
        cond = True
        for k in kernelfields:
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


    def predict(self, query, op, target_freq, max_entries=20, \
                target_gpu_config=None, target_dvfs_supply_voltage=None, target_dvfs_idle_power=None, \
                use_precomputed_coeffs=False, ignore_exact=False, verbose=False, **kwargs):
        if (op == 'softmax') or (op == 'layernorm') :
            assert(set(['batch', 'dim', 'prec']).issubset(list(query.keys())))
        elif (op=='softmax_fusion') or (op=='layernorm_fusion'):
            assert(set(['batch', 'dim', 'prec', 'fusion_ignore_in', 'fusion_ignore_out'])).issubset(list(query.keys()))
        else:
            raise NotImplementedError()

        self.predict_kernel(query, op)

        if verbose:
            print("Predicted kernel information ---")
            for k, v in query.items():
                print(k, v)

        if target_freq is None:
            query['avg_freq'] = 1000
        else:
            query['avg_freq'] = target_freq

        # Fusion cases
        if (op=='softmax_fusion') or (op=='layernorm_fusion'):
            adjust_gmem_to_sm_ratio = 0 if query['fusion_ignore_in'] else 1
            adjust_sm_to_gmem_ratio = 0 if query['fusion_ignore_out'] else 1
            fusion = True
            op = op.split('_')[0] # softmax or layernorm
        else:
            adjust_gmem_to_sm_ratio = 1
            adjust_sm_to_gmem_ratio = 1
            fusion = False

        query = pd.DataFrame([query], index=[0])
        if target_gpu_config is not None:
            target_gpu_analytical_model = NonlinearAnalyticalModel(target_gpu_config, \
                                                                   dvfs_supply_voltage=target_dvfs_supply_voltage, \
                                                                   dvfs_idle_power=target_dvfs_idle_power)
            target_gpu_analytical_model.model(query, op, False, adjust_gmem_to_sm_ratio=adjust_gmem_to_sm_ratio, adjust_sm_to_gmem_ratio=adjust_sm_to_gmem_ratio)
            del target_gpu_analytical_model
        else:
            if self.multiple_configs:
                print("Error: multiple GPU configuration in the database is only supported for extrapolation cases!")
                exit()
            self.analytical_model.model(query, op, False, adjust_gmem_to_sm_ratio=adjust_gmem_to_sm_ratio, adjust_sm_to_gmem_ratio=adjust_sm_to_gmem_ratio)
        query = dict(query.loc[0])

        precompute_success = False
        precomputed_coeffs = None
        if use_precomputed_coeffs:
            flag = 'precompute'
            precomputed_coeffs = self._get_precomputed_coeffs(query, op)
            ref = None

            perfcoeffs = precomputed_coeffs[0]
            powercoeffs = precomputed_coeffs[1]
            if (type(perfcoeffs) == str):
                perfcoeffs = eval(perfcoeffs[1:-1])
            
            if (type(powercoeffs)== str):
                powercoeffs = eval(powercoeffs[1:-1])

            precomputed_coeffs = (perfcoeffs, powercoeffs)

            if ((precomputed_coeffs[0]) == -1):
                precompute_success = False
            else:
                precompute_success = True

        if (not use_precomputed_coeffs) or (not precompute_success):

            flag, ref = self.get_references(query, op, max_entries=max_entries, ignore_exact=(self.dvfs_aware or fusion or ignore_exact))

            # Empty target frequency -> only for non-multiple config cases
            # TODO: Add warning
            if (target_freq is None) and (flag == 'ref'):
                target_freq = ref['avg_freq'].mean()
                query['avg_freq'] = target_freq
                query = pd.DataFrame([query], index=[0])
                self.analytical_model.model(query, op, train=False, adjust_gmem_to_sm_ratio=adjust_gmem_to_sm_ratio, adjust_sm_to_gmem_ratio=adjust_sm_to_gmem_ratio)
                query = dict(query.loc[0])

            if verbose:
                print("Reference status: {} | Reference entry size: {}".format(flag, len(ref)))
                # print(ref.to_markdown())
                
            # DVFS Inference Mode is all, then use all sources
            if self.dvfs_inference_mode == 'all':
                dfs_ref = []
                prec = query['prec']
                for dvfs_df in self.model_database[(op, prec)]:
                    _df = pd.merge(dvfs_df, ref, on=['batch', 'dim'], how='inner', indicator=True, suffixes=[None, 'ref'])
                    _df = _df.loc[:, dvfs_df.columns]
                    dfs_ref.append(_df)
                ref = pd.concat(dfs_ref, ignore_index=True)

        if flag == 'exact':
            return (ref.iloc[0]['time'], -1, ref.iloc[0]['energy'])
        else:
            estimated_time, coeffs = self._predict_time(query, ref, fusion=fusion, \
                                                        use_precomputed_coeffs=(use_precomputed_coeffs and precompute_success), precomputed_coeffs=precomputed_coeffs, \
                                                        verbose=verbose)
            estimated_power = self._predict_power(query, ref, coeffs, \
                                                  dvfs=self.dvfs_aware, fusion=fusion, \
                                                  use_precomputed_coeffs=(use_precomputed_coeffs and precompute_success), precomputed_coeffs=precomputed_coeffs, \
                                                  verbose=verbose)
            if type(estimated_power) == pd.core.series.Series:
                estimated_power = estimated_power.values[0]

            power_cap = self.gpu_config['power_cap'] if target_gpu_config is None else target_gpu_config['power_cap']
            power = min(estimated_power, power_cap)
            estimated_energy = power * estimated_time / 1000.
            return (estimated_time, estimated_power, estimated_energy)


    
    