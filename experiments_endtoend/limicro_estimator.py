import sys
import os

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import yaml

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier

try:
    from .estimator import Estimator
except:
    from estimator import Estimator

sys.path.append('..')
from gee import get_gee
from gee.frontend_utils import *

class LiEstimator(Estimator):
    def __init__(self, gpu_config_yaml, lut_config_yaml, lut_folder_abs_path='/mnt/c/Users/KyungmiLee/Documents/gpu-energy-estimation/lut', **kwargs):
        super().__init__(gpu_config_yaml, lut_config_yaml, lut_folder_abs_path, **kwargs)

        self.predictor = {}
        self.kernel_predictor = {}

        self.gee = get_gee(gpu_yaml_path=gpu_config_yaml, \
                           lut_yaml_path=lut_config_yaml, \
                           lut_folder_abs_path=lut_folder_abs_path, \
                           no_build=True)

        with open(gpu_config_yaml, 'r') as f:
            self.gpu_config = yaml.safe_load(f)['gpu_configs']
        with open(lut_config_yaml, 'r') as f:
            self.lut_config = yaml.safe_load(f)['lut_config']

        # Build predictor
        if 'gemm' in self.lut_config.keys():
            for key, value in self.lut_config['gemm'].items():
                for kkey, vvalue in self.lut_config['gemm'][key].items():
                    op_type = ('gemm', key, kkey)
                    for entry in vvalue:
                        if entry['main']:
                            df = pd.read_csv(os.path.join(lut_folder_abs_path, entry['path']))
                            self.predictor[op_type] = self._build_predictor(df, 'gemm')
                            self.kernel_predictor[op_type] = self._build_kernel_predictor(df, op_type)
                            break

        if 'conv2d' in self.lut_config.keys():
            for key, value in self.lut_config['conv2d'].items():
                for kkey, vvalue in self.lut_config['conv2d'][key].items():
                    op_type = ('conv2d', key, kkey)
                    for entry in vvalue:
                        if entry['main']:
                            df = pd.read_csv(os.path.join(lut_folder_abs_path, entry['path']))
                            self.predictor[op_type] = self._build_predictor(df, 'conv2d')
                            self.kernel_predictor[op_type] = self._build_kernel_predictor(df, op_type)
                            break
                    
        if 'softmax' in self.lut_config.keys():
            for key, value in self.lut_config['softmax'].items():
                op_type = ('softmax', key)
                for entry in value:
                    if entry['main']:
                        df = pd.read_csv(os.path.join(lut_folder_abs_path, entry['path']))
                        self.predictor[op_type] = self._build_predictor(df, 'softmax')
                        self.kernel_predictor[op_type] = self._build_kernel_predictor(df, op_type)
                        break

        if 'layernorm' in self.lut_config.keys():
            for key, value in self.lut_config['layernorm'].items():
                op_type = ('layernorm', key)
                for entry in value:
                    if entry['main']:
                        df = pd.read_csv(os.path.join(lut_folder_abs_path, entry['path']))
                        self.predictor[op_type] = self._build_predictor(df, 'layernorm')
                        self.kernel_predictor[op_type] = self._build_kernel_predictor(df, op_type)
                        break

        if 'elementwise' in self.lut_config.keys():
            op_type = ('elementwise',)
            for entry in self.lut_config['elementwise']:
                if entry['main']:
                    df = pd.read_csv(os.path.join(lut_folder_abs_path, entry['path']))
                    self.predictor[op_type] = self._build_predictor(df, 'elementwise')
                    self.kernel_predictor[op_type] = self._build_kernel_predictor(df, op_type)
                    break

        
    def _build_predictor(self, df, op_type):
        time_regressors = self._build_predictor_for_target(df, op_type, 'time')
        energy_regressors = self._build_predictor_for_target(df, op_type, 'energy')

        return (time_regressors, energy_regressors)
    
    def _elementwise_get_num_input_args(self, op):
        if op in ['pointwise_mul', 'pointwise_add', 'unspecified_tensor']:
            return 2
        else:
            return 1

    def _build_predictor_for_target(self, df, op_type, target):

        # For Conv2D, drop too large workloads for LiMicro (very sensitive to data distribution - too large or too small workloads distort the linear regression, causing high error)
        if op_type == 'conv2d':
            df = df.loc[(df['dimM'] < 2 ** 17) & (df['dimN'] < 2 ** 17) & (df['dimK'] < 2 ** 17)].reset_index(drop=True)

        kernels = df['kernel_name'].unique()
        regressors = {}

        for kernel in kernels:
            # Check if df_train has this kernel
            sub_df = df.loc[df['kernel_name'] == kernel]
            n_data = len(sub_df)

            if n_data == 0:
                continue

            if n_data == 1:
                slope = 0.0
                intercept = sub_df[target].values[0]
                regressors[kernel] = (slope, intercept, 'constant', 'normal')
                continue

            if (op_type == 'gemm'):
                sub_df['flop'] = sub_df['batch'] * sub_df['dimM'] * sub_df['dimN'] * sub_df['dimK']
                sub_df['input_size'] = sub_df['batch'] * (sub_df['dimM'] * sub_df['dimK'] + sub_df['dimN'] * sub_df['dimK'])
                sub_df['output_size'] = sub_df['batch'] * sub_df['dimM'] * sub_df['dimN']
            elif (op_type == 'conv2d'):
                sub_df['flop'] = sub_df['batch'] * sub_df['dimM'] * sub_df['dimN'] * sub_df['dimK']
                sub_df['input_size'] = sub_df['b'] * sub_df['c'] * sub_df['hw'] * sub_df['hw']
                sub_df['pq'] = np.floor((sub_df['hw'] - sub_df['rs'] + 2 * sub_df['padding']) / sub_df['stride']) + 1
                sub_df['output_size'] = sub_df['b'] * sub_df['m'] * sub_df['pq'] * sub_df['pq']
            elif (op_type == 'softmax') or (op_type == 'layernorm'):
                sub_df['flop'] = sub_df['batch'] * sub_df['dim']
                sub_df['input_size'] = sub_df['batch'] * sub_df['dim']
                sub_df['output_size'] = sub_df['batch'] * sub_df['dim']
            elif (op_type == 'elementwise'):
                sub_df['flop'] = sub_df['dim']
                sub_df['n_input_args'] = sub_df['op'].apply(lambda x : self._elementwise_get_num_input_args(x))
                sub_df['input_size'] = sub_df['dim'] * sub_df['n_input_args']
                sub_df['output_size'] = sub_df['dim']

            if n_data < 10:
                # No train, test split
                
                # flop
                X, y = sub_df['flop'].values.reshape(-1, 1), sub_df[target].values.reshape(-1, 1)
                reg1 = LinearRegression().fit(X, y)
                score1 = reg1.score(X, y)

                # input
                X, y = sub_df['input_size'].values.reshape(-1, 1), sub_df[target].values.reshape(-1, 1)
                reg2 = LinearRegression().fit(X, y)
                score2 = reg2.score(X, y)

                # output
                X, y = sub_df['output_size'].values.reshape(-1, 1), sub_df[target].values.reshape(-1, 1)
                reg3 = LinearRegression().fit(X, y)
                score3 = reg3.score(X, y)

                # flop - log/log
                X, y = np.log10(sub_df['flop'].values.reshape(-1, 1)), np.log10(sub_df[target].values.reshape(-1, 1))
                reg4 = LinearRegression().fit(X, y)
                score4 = reg4.score(X, y)

                # input - log/log
                X, y = np.log10(sub_df['input_size'].values.reshape(-1, 1)), np.log10(sub_df[target].values.reshape(-1, 1))
                reg5 = LinearRegression().fit(X, y)
                score5 = reg5.score(X, y)

                # output - log/log
                X, y = np.log10(sub_df['output_size'].values.reshape(-1, 1)), np.log10(sub_df[target].values.reshape(-1, 1))
                reg6 = LinearRegression().fit(X, y)
                score6 = reg6.score(X, y)

                max_score = max(score1, score2, score3, score4, score5, score6)
                if max_score == score1:
                    regressors[kernel] = (reg1.coef_[0], reg1.intercept_, 'flop', 'normal')
                elif max_score == score2:
                    regressors[kernel] = (reg2.coef_[0], reg2.intercept_, 'input', 'normal')
                elif max_score == score3:
                    regressors[kernel] = (reg3.coef_[0], reg3.intercept_, 'output', 'normal')
                elif max_score == score4:
                    regressors[kernel] = (reg4.coef_[0], reg4.intercept_, 'flop', 'log')
                elif max_score == score5:
                    regressors[kernel] = (reg5.coef_[0], reg5.intercept_, 'input', 'log')
                else:
                    regressors[kernel] = (reg6.coef_[0], reg6.intercept_, 'output', 'log')
                
            else:
                # Train, test split

                # flop
                X, y = sub_df['flop'].values.reshape(-1, 1), sub_df[target].values.reshape(-1, 1)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
                reg1 = LinearRegression().fit(X_train, y_train)
                score1 = reg1.score(X_test, y_test)

                # input
                X, y = sub_df['input_size'].values.reshape(-1, 1), sub_df[target].values.reshape(-1, 1)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
                reg2 = LinearRegression().fit(X_train, y_train)
                score2 = reg2.score(X_test, y_test)

                # output
                X, y = sub_df['output_size'].values.reshape(-1, 1), sub_df[target].values.reshape(-1, 1)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
                reg3 = LinearRegression().fit(X_train, y_train)
                score3 = reg3.score(X_test, y_test)

                # flop - log/log
                X, y = np.log10(sub_df['flop'].values.reshape(-1, 1)), np.log10(sub_df[target].values.reshape(-1, 1))
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
                reg4 = LinearRegression().fit(X_train, y_train)
                score4 = reg4.score(X_test, y_test)

                # input - log/log
                X, y = np.log10(sub_df['input_size'].values.reshape(-1, 1)), np.log10(sub_df[target].values.reshape(-1, 1))
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
                reg5 = LinearRegression().fit(X_train, y_train)
                score5 = reg5.score(X_test, y_test)

                # output - log/log
                X, y = np.log10(sub_df['output_size'].values.reshape(-1, 1)), np.log10(sub_df[target].values.reshape(-1, 1))
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
                reg6 = LinearRegression().fit(X_train, y_train)
                score6 = reg6.score(X_test, y_test)

                max_score = max(score1, score2, score3, score4, score5, score6)
                if max_score == score1:
                    regressors[kernel] = (reg1.coef_[0], reg1.intercept_, 'flop', 'normal')
                elif max_score == score2:
                    regressors[kernel] = (reg2.coef_[0], reg2.intercept_, 'input', 'normal')
                elif max_score == score3:
                    regressors[kernel] = (reg3.coef_[0], reg3.intercept_, 'output', 'normal')
                elif max_score == score4:
                    regressors[kernel] = (reg4.coef_[0], reg4.intercept_, 'flop', 'log')
                elif max_score == score5:
                    regressors[kernel] = (reg5.coef_[0], reg5.intercept_, 'input', 'log')
                else:
                    regressors[kernel] = (reg6.coef_[0], reg6.intercept_, 'output', 'log')

        return regressors
    
    def _build_kernel_predictor(self, df, op_type):

        # For Conv2D, drop too large workloads for LiMicro (very sensitive to data distribution - too large or too small workloads distort the linear regression, causing high error)
        if op_type[0] == 'conv2d':
            df = df.loc[(df['dimM'] < 2 ** 17) & (df['dimN'] < 2 ** 17) & (df['dimK'] < 2 ** 17)].reset_index(drop=True)

        kernels = list(df['kernel_name'].unique())
        kernel_map = {idx: value for idx, value in enumerate(kernels)}

        df['kernel_id'] = df['kernel_name'].apply(lambda x: kernels.index(x))

        if (op_type[0] == 'gemm'):
            weak_learner = tree.DecisionTreeClassifier(max_depth=15, ccp_alpha=0.0005)
            kernel_predictor = AdaBoostClassifier(estimator=weak_learner, n_estimators=10)
            kernel_predictor = kernel_predictor.fit(df[['batch', 'dimM', 'dimN', 'dimK']], df['kernel_id'])
        elif (op_type[0] == 'conv2d'):
            weak_learner = tree.DecisionTreeClassifier(max_depth=15, ccp_alpha=0.0005)
            kernel_predictor = AdaBoostClassifier(estimator=weak_learner, n_estimators=10)
            kernel_predictor = kernel_predictor.fit(df[['b','m','c','hw','rs','stride','padding']], df['kernel_id'])
        elif (op_type[0] == 'softmax') or (op_type[0] == 'layernorm'):
            kernel_predictor = tree.DecisionTreeClassifier()
            kernel_predictor = kernel_predictor.fit(df[['batch', 'dim']], df['kernel_id'])
        elif (op_type[0] == 'elementwise'):
            kernel_predictor = {}
            unique_ops = list(df['op'].unique())
            for op in unique_ops:
                unique_prec = list(df.loc[df['op'] == op]['prec'].unique())
                for p in unique_prec:
                    _df = df.loc[(df['op'] == op) & (df['prec'] == p)]
                    kernel_predictor[(op, p)] = _df['kernel_id'].mode()[0]
        
        return (kernel_map, kernel_predictor)
    
    def _predict_single_kernel_for_target(self, query, regressor, op_type):
        slope = regressor[0]
        intercept = regressor[1]
        input_type = regressor[2]
        log = (regressor[3] == 'log')

        if input_type == 'constant':
            estimated = intercept
        
        elif input_type == 'flop':
            if (op_type == 'gemm') or (op_type == 'conv2d'):
                x = query['batch'] * query['dimM'] * query['dimN'] * query['dimK']
            elif (op_type == 'softmax') or (op_type == 'layernorm'):
                x = query['batch'] * query['dim']
            elif (op_type == 'elementwise'):
                x = query['dim']

            if log:
                x = np.log10(x)
            estimated = x * slope + intercept
            if log:
                estimated = 10 ** estimated

        elif input_type == 'input':
            if (op_type == 'gemm'):
                x = query['batch'] * (query['dimM'] * query['dimK'] + query['dimN'] * query['dimK'])
            elif (op_type == 'conv2d'):
                x = query['b'] * query['c'] * query['hw'] * query['hw'] + query['m'] * query['c'] * query['rs'] * query['rs']
            elif (op_type == 'softmax') or (op_type == 'layernorm'):
                x = query['batch'] * query['dim']
            elif (op_type == 'elementwise'):
                x = query['dim'] * self._elementwise_get_num_input_args(query['op'])

            if log:
                x = np.log10(x)
            estimated = x * slope + intercept
            if log:
                estimated = 10 ** estimated

        elif input_type == 'output':
            if (op_type == 'gemm'):
                x = query['batch'] * query['dimM'] * query['dimN']
            elif (op_type == 'conv2d'):
                pq = np.floor((query['hw'] - query['rs'] + 2 * query['padding']) / query['stride']) + 1
                x = query['b'] * query['m'] * pq * pq
            elif (op_type == 'softmax') or (op_type == 'layernorm'):
                x = query['batch'] * query['dim']
            elif (op_type == 'elementwise'):
                x = query['dim']
                
            if log:
                x = np.log10(x)
            estimated = x * slope + intercept
            if log:
                estimated = 10 ** estimated

        return estimated

    def predict_single_kernel(self, query, op_type, **kwargs):
        # predict kernel from query
        kernel_map, kernel_predictor = self.kernel_predictor[op_type]

        if (op_type[0] == 'gemm'):
            kernel = kernel_map[kernel_predictor.predict([[query['batch'], query['dimM'], query['dimN'], query['dimK']]])[0]]
        elif (op_type[0] == 'conv2d'):
            kernel = kernel_map[kernel_predictor.predict([[query['b'], query['m'], query['c'], query['hw'], query['rs'], query['stride'], query['padding']]])[0]]
        elif (op_type[0] == 'softmax') or (op_type[0] == 'layernorm'):
            kernel = kernel_map[kernel_predictor.predict([[query['batch'], query['dim']]])[0]]
        elif (op_type[0] == 'elementwise'):
            # kernel_predictor_keys = list(set([x[0] for x in kernel_predictor.keys()]))
            if query['op'] == 'unspecified_activation':
                query['op'] = 'relu' # as default
            elif query['op'] == 'unspecified_tensor':
                query['op'] = 'pointwise_add'
            elif query['op'] == 'unspecified_scalar':
                query['op'] = 'scalar_add'
            kernel = kernel_map[kernel_predictor[(query['op'], query['prec'])]]

        # regressor
        time_regressors, energy_regressors = self.predictor[op_type]

        regressor = time_regressors[kernel]
        estimated_time = self._predict_single_kernel_for_target(query, regressor, op_type[0])

        regressor = energy_regressors[kernel]
        estimated_energy = self._predict_single_kernel_for_target(query, regressor, op_type[0])

        try:
            estimated_time = estimated_time[0]
        except:
            pass

        try:
            estimated_energy = estimated_energy[0]
        except:
            pass

        return (estimated_time, estimated_energy)

    def lookup_einsum(self, einsum_args, precM, precA, use_tensorcore, target_freq=None, lookup_target='energy', verbose=False, \
                      kernel_info=None, transpose_mn=False):
        
        gemm_list = parse_einsum(einsum_args, self.gee.einsum_parse_cache, self.gee.enable_einsum_cache, False)

        if target_freq is None:
            target_freq = self.gpu_config['sm_max_freq']

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

            if verbose:
                print("Resolved to : batch {} | M {} | N {} | K {}".format(resultvars['batch'], resultvars['dimM'], resultvars['dimN'], resultvars['dimK']))
            
            query_type = ('gemm', 'tc' if use_tensorcore else 'cuda', '{}_{}'.format(precM, precA))
            # estimated = self.lookup(resultvars, query_type, target_freq, verbose=verbose, lookup_target=lookup_target, kernel_info_provided=kernel_info_provided)
            estimated = self.predict_single_kernel(resultvars, query_type)
            
            if lookup_target == 'time':
                energy_list.append(estimated[0])
            elif lookup_target == 'energy':
                energy_list.append(estimated[1])
            else:
                energy_list.append((estimated[0], -1, estimated[1]))
            
        return energy_list