import os
import sys

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

import tqdm

class Estimator():
    def __init__(self, train_data_csv, test_data_csv, **kwargs):
        self.train_data_csv = train_data_csv
        self.test_data_csv = test_data_csv

        # self.train_df = pd.read_csv(self.train_data_csv)
        # self.test_df = pd.read_csv(self.test_data_csv)

        self.predictor = {}
        self.trained = {'energy': False, 'time': False}

        # Preprocess data if necessary
        # self.preprocess()

    def reset_predictor(self):
        self.predictor = {}
        self.trained = {'energy': False, 'time': False}

    def reset_testdf(self):
        self.test_df = pd.read_csv(self.test_data_csv)

    def preprocess(self):
        raise NotImplementedError()
    
    def train(self, target):
        raise NotImplementedError()
    
    def test(self, target):
        raise NotImplementedError()
    
    def generate_test_report(self, test_for, save_to, print_summary=True):
        for target in test_for:
            if self.trained[target] == False:
                self.train(target)
            self.test(target)

            # Drop any entries with estimated energy/time is -1 (no prediction available for li, micro'23 paper)
            self.test_df.drop(self.test_df[self.test_df['{}_estimate'.format(target)] < 0].index, inplace=True)
            self.test_df.reset_index(inplace=True, drop=True)
            # self.test_df[self.test_df['{}_estimate'.format(target)] > 0].reset_index(drop=True, inplace=True)

            self.test_df['{}_error'.format(target)] = self.test_df[target] - self.test_df['{}_estimate'.format(target)]
            self.test_df['{}_percent_error'.format(target)] = self.test_df['{}_error'.format(target)] / self.test_df[target] * 100.
            # print(self.test_df.columns)

        # Dump the test result
        self.test_df.to_csv(save_to, index=False)

        # Print summary: MAPE, Absolute Error 
        if print_summary:
            for target in test_for:
                error = self.test_df['{}_error'.format(target)].values
                error_pct = self.test_df['{}_percent_error'.format(target)].values

                mean_abs_error = np.abs(error).mean()
                mean_abs_percent_error = np.abs(error_pct).mean()

                print("================ Prediction Target: {} ================".format(target))
                print("Average absolute error: {:.4f} {}".format(mean_abs_error, 'J' if target == 'energy' else 'ms'))
                print("Average absolute percent error: {:.4f} %".format(mean_abs_percent_error))
                # print("=======================================================")