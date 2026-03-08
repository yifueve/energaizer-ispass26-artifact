import os
import sys

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from estimator import Estimator

class LiEstimator(Estimator):
    def __init__(self, train_data_csv, test_data_csv, **kwargs):
        super().__init__(train_data_csv, test_data_csv, **kwargs)

        # Load train df
        self.train_df = pd.read_csv(train_data_csv)
        self.workload_type = kwargs['workload_type']
        self.estimation_target = kwargs['estimation_target'] # throughput or latency (only for time)

        self.preprocess()

    def preprocess(self):
        self.reset_testdf()
        # No other preprocessing needed

    def _elementwise_get_num_input_args(self, op):
        if op in ['pointwise_mul', 'pointwise_add', 'unspecified_tensor']:
            return 2
        else:
            return 1

    def train(self, target, verbose=False):

        def get_y(sub_df, log=False, target=target):
            if (self.estimation_target == 'throughput') and target == 'time':
                y = sub_df['flop'].values.reshape(-1, 1) / sub_df[target].values.reshape(-1, 1)
            else:
                y = sub_df[target].values.reshape(-1, 1)

            if log:
                y = np.log10(y)
            
            return y
                
        assert (target in ['time', 'energy'])

        # Train a per-kernel linear regression
        kernels = self.train_df['kernel_name'].unique()
        regressors = {} # For each kernel, save slope, intercept, input type

        for kernel in kernels:
            # Check if df_train has this kernel
            sub_df = self.train_df.loc[self.train_df['kernel_name'] == kernel]
            n_data = len(sub_df)

            if verbose:
                print("Processing kernel: {}".format(kernel))

            if n_data == 0:
                if verbose:
                    print("- No data for this kernel in train dataset. Skip.")
                continue

            if n_data == 1:
                if verbose:
                    print("- Just a single data point for this kernel. Constant energy.")
                slope = 0.0
                intercept = sub_df[target].values[0]
                regressors[kernel] = (slope, intercept, 'constant', 'normal')
                continue

            if (self.workload_type == 'gemm'):
                sub_df['flop'] = sub_df['batch'] * sub_df['dimM'] * sub_df['dimN'] * sub_df['dimK']
                sub_df['input_size'] = sub_df['batch'] * (sub_df['dimM'] * sub_df['dimK'] + sub_df['dimN'] * sub_df['dimK'])
                sub_df['output_size'] = sub_df['batch'] * sub_df['dimM'] * sub_df['dimN']
            elif (self.workload_type == 'conv2d'):
                sub_df['flop'] = sub_df['batch'] * sub_df['dimM'] * sub_df['dimN'] * sub_df['dimK']
                sub_df['input_size'] = sub_df['b'] * sub_df['c'] * sub_df['hw'] * sub_df['hw']
                sub_df['pq'] = np.floor((sub_df['hw'] - sub_df['rs'] + 2 * sub_df['padding']) / sub_df['stride']) + 1
                sub_df['output_size'] = sub_df['b'] * sub_df['m'] * sub_df['pq'] * sub_df['pq']
            elif (self.workload_type == 'softmax') or (self.workload_type == 'layernorm'):
                sub_df['flop'] = sub_df['batch'] * sub_df['dim']
                sub_df['input_size'] = sub_df['batch'] * sub_df['dim']
                sub_df['output_size'] = sub_df['batch'] * sub_df['dim']
            elif (self.workload_type == 'elementwise'):
                sub_df['flop'] = sub_df['dim']
                sub_df['n_input_args'] = sub_df['op'].apply(lambda x : self._elementwise_get_num_input_args(x))
                sub_df['input_size'] = sub_df['dim'] * sub_df['n_input_args']
                sub_df['output_size'] = sub_df['dim']

            

            if n_data < 10:
                # No train, test split
                
                # flop
                X = sub_df['flop'].values.reshape(-1, 1)
                y = get_y(sub_df)
                reg1 = LinearRegression().fit(X, y)
                score1 = reg1.score(X, y)

                # input
                X = sub_df['input_size'].values.reshape(-1, 1)
                y = get_y(sub_df)
                reg2 = LinearRegression().fit(X, y)
                score2 = reg2.score(X, y)

                # output
                X = sub_df['output_size'].values.reshape(-1, 1)
                y = get_y(sub_df)
                reg3 = LinearRegression().fit(X, y)
                score3 = reg3.score(X, y)

                # flop - log/log
                X = np.log10(sub_df['flop'].values.reshape(-1, 1))
                y = get_y(sub_df, log=True)
                reg4 = LinearRegression().fit(X, y)
                score4 = reg4.score(X, y)

                # input - log/log
                X = np.log10(sub_df['input_size'].values.reshape(-1, 1))
                y = get_y(sub_df, log=True)
                reg5 = LinearRegression().fit(X, y)
                score5 = reg5.score(X, y)

                # output - log/log
                X = np.log10(sub_df['output_size'].values.reshape(-1, 1))
                y = get_y(sub_df, log=True)
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
                X = sub_df['flop'].values.reshape(-1, 1)
                y = get_y(sub_df)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
                reg1 = LinearRegression().fit(X_train, y_train)
                score1 = reg1.score(X_test, y_test)

                # input
                X = sub_df['input_size'].values.reshape(-1, 1)
                y = get_y(sub_df)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
                reg2 = LinearRegression().fit(X_train, y_train)
                score2 = reg2.score(X_test, y_test)

                # output
                X = sub_df['output_size'].values.reshape(-1, 1)
                y = get_y(sub_df)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
                reg3 = LinearRegression().fit(X_train, y_train)
                score3 = reg3.score(X_test, y_test)

                # flop - log/log
                X = np.log10(sub_df['flop'].values.reshape(-1, 1))
                y = get_y(sub_df, log=True)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
                reg4 = LinearRegression().fit(X_train, y_train)
                score4 = reg4.score(X_test, y_test)

                # input - log/log
                X = np.log10(sub_df['input_size'].values.reshape(-1, 1))
                y = get_y(sub_df, log=True)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
                reg5 = LinearRegression().fit(X_train, y_train)
                score5 = reg5.score(X_test, y_test)

                # output - log/log
                X = np.log10(sub_df['output_size'].values.reshape(-1, 1))
                y = get_y(sub_df, log=True)
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

        
        self.predictor[target] = regressors
        self.trained[target] = True

    def test(self, target):
        # Test on the test df for target
        self.test_df['{}_estimate'.format(target)] = -1
        regressors = self.predictor[target]

        for i, row in self.test_df.iterrows():
            kernel = row['kernel_name']
            if kernel in regressors.keys():
                slope = regressors[kernel][0]
                intercept = regressors[kernel][1]
                input_type = regressors[kernel][2]
                log = (regressors[kernel][3] == 'log')

                if input_type == 'constant':
                    estimated = intercept
                
                elif input_type == 'flop':
                    if (self.workload_type == 'gemm') or (self.workload_type ==  'conv2d'):
                        x = row['batch'] * row['dimM'] * row['dimN'] * row['dimK']
                    elif (self.workload_type == 'softmax') or (self.workload_type == 'layernorm'):
                        x = row['batch'] * row['dim']
                    elif (self.workload_type == 'elementwise'):
                        x = row['dim']

                    if log:
                        x = np.log10(x)
                    estimated = x * slope + intercept
                    if log:
                        estimated = 10 ** estimated

                elif input_type == 'input':
                    if (self.workload_type == 'gemm'):
                        x = row['batch'] * (row['dimM'] * row['dimK'] + row['dimN'] * row['dimK'])
                    elif (self.workload_type == 'conv2d'):
                        x = row['b'] * row['c'] * row['hw'] * row['hw'] + row['m'] * row['c'] * row['rs'] * row['rs']
                    elif (self.workload_type == 'softmax') or (self.workload_type == 'layernorm'):
                        x = row['batch'] * row['dim']
                    elif (self.workload_type == 'elementwise'):
                        x = row['dim'] * self._elementwise_get_num_input_args(row['op'])

                    if log:
                        x = np.log10(x)
                    estimated = x * slope + intercept
                    if log:
                        estimated = 10 ** estimated

                elif input_type == 'output':
                    if (self.workload_type == 'gemm'):
                        x = row['batch'] * row['dimM'] * row['dimN']
                    elif (self.workload_type == 'conv2d'):
                        pq = np.floor((row['hw'] - row['rs'] + 2 * row['padding']) / row['stride']) + 1
                        x = row['b'] * row['m'] * pq * pq
                    elif (self.workload_type == 'softmax') or (self.workload_type == 'layernorm'):
                        x = row['batch'] * row['dim']
                    elif (self.workload_type == 'elementwise'):
                        x = row['dim']

                    if log:
                        x = np.log10(x)
                    estimated = x * slope + intercept
                    if log:
                        estimated = 10 ** estimated

                else:
                    print("Not supported type! Check your code! Error type: {}, {}".format(kernel, input_type))
                    continue
            
            else:
                print("No train data point for this kernel {}!".format(kernel))
                estimated = -1

            if (self.estimation_target == 'throughput') and (target == 'time'):
                if (self.workload_type == 'gemm'):
                    x = row['batch'] * (row['dimM'] * row['dimK'] + row['dimN'] * row['dimK'])
                elif (self.workload_type == 'conv2d'):
                    x = row['b'] * row['c'] * row['hw'] * row['hw'] + row['m'] * row['c'] * row['rs'] * row['rs']
                elif (self.workload_type == 'softmax') or (self.workload_type == 'layernorm'):
                    x = row['batch'] * row['dim']
                elif (self.workload_type == 'elementwise'):
                    x = row['dim'] * self._elementwise_get_num_input_args(row['op'])
                estimated = x / estimated

            self.test_df.loc[i, '{}_estimate'.format(target)] = estimated