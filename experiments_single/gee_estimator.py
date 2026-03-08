import os
import sys

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import yaml

sys.path.append('..')
from gee import get_gee

from estimator import Estimator

from tqdm import tqdm

class GeeEstimator(Estimator):
    def __init__(self, train_data_csv, test_data_csv, **kwargs):
        super().__init__(train_data_csv, test_data_csv, **kwargs)

        # Gee initialization
        self.gee = get_gee(gpu_yaml_path=kwargs['gpu_config'], \
                           lut_yaml_path=kwargs['lut_config'], \
                           use_entire_references_for_estimate=kwargs['use_entire_ref'], \
                           dvfs_aware=kwargs['dvfs_aware'], \
                           dvfs_inference_mode='single_source', \
                           dvfs_idle_power_json=kwargs['dvfs_idle_power_json'], \
                           dvfs_supply_voltage_json=kwargs['dvfs_supply_voltage_json'], \
                           lut_folder_abs_path=os.path.realpath(os.path.join(kwargs['lut_parent_path'], 'lut')))
        
        # Assert that the path in lut_config should match with train_data_csv
        with open(kwargs['lut_config']) as f:
            try:
                lut_config = yaml.safe_load(f)['lut_config']
            except yaml.YAMLError as exc:
                print(exc)
                print("Error while loading gpu configuration yaml file")
                exit()

        self.workload_type = kwargs['workload_type']

        # LUT Checking
        if self.workload_type == 'gemm':
            assert (len(lut_config['gemm']) == 1) # either cuda or tc
            for key, value in lut_config['gemm'].items():
                self.op_type = key + '_gemm'
                assert(len(value) == 1) # only one lut for this
                for kkey, vvalue in value.items():
                    self.prec_type = kkey
                    # path is vvalue
                    path = os.path.realpath(os.path.join(kwargs['lut_parent_path'], 'lut', vvalue[0]['path']))
                    assert(path == os.path.realpath(self.train_data_csv))

        elif self.workload_type == 'conv2d':
            assert (len(lut_config['conv2d']) == 1) # either cuda or tc
            for key, value in lut_config['conv2d'].items():
                self.op_type = ('conv2d', key)
                assert(len(value) == 1) # only one lut for this
                for kkey, vvalue in value.items():
                    self.prec_type = kkey
                    # path is vvalue
                    path = os.path.realpath(os.path.join(kwargs['lut_parent_path'], 'lut', vvalue[0]['path']))
                    assert(path == os.path.realpath(self.train_data_csv))
        
        elif (self.workload_type == 'softmax') or (self.workload_type == 'layernorm'):
            assert (len(lut_config[self.workload_type]) == 1) # one precision
            for key, value in lut_config[self.workload_type].items():
                self.op_type = self.workload_type
                self.prec_type = key
                path = os.path.realpath(os.path.join(kwargs['lut_parent_path'], 'lut', value[0]['path']))
                assert(path == os.path.realpath(self.train_data_csv))

        elif (self.workload_type == 'elementwise'):
            assert (len(lut_config[self.workload_type]) == 1) # one lut
            for elem in lut_config[self.workload_type]:
                self.op_type = self.workload_type
                path = os.path.realpath(os.path.join(kwargs['lut_parent_path'], 'lut', elem['path']))
                assert(path == os.path.realpath(self.train_data_csv))


        self.trained = {'energy': True, 'time': True}

        self.preprocess()

    def preprocess(self):
        if self.test_data_csv != '':
            self.reset_testdf()

    def train(self, target):
        assert (target in ['time', 'energy'])
        self.trained[target] = True
        # self.predictor[target] = self.gee
        # For the Gee LUT approach, no training needed after initialization

    def test(self, target):
        # Test on the test df for target
        self.test_df['{}_estimate'.format(target)] = -1

        for i, row in tqdm(self.test_df.iterrows()):

            if self.workload_type == 'gemm':
                query = {'batch': row['batch'], \
                        'dimM': row['dimM'], \
                        'dimN': row['dimN'], \
                        'dimK': row['dimK'], \
                        'trans': row['trans'], \
                        'precM': row['precM'], \
                        'precA': row['precA'], \
                        'useTensorCore': row['useTensorCore']}
                query_type = (self.op_type, self.prec_type)
            elif (self.workload_type == 'softmax') or (self.workload_type == 'layernorm'):
                query = {'batch': row['batch'], \
                         'dim': row['dim'], \
                         'prec': row['prec']}
                query_type = (self.op_type, self.prec_type)
            elif self.workload_type == 'conv2d':
                query = {'batch': row['batch'], \
                         'dimM': row['dimM'], \
                         'dimN': row['dimN'], \
                         'dimK': row['dimK'], \
                         'b': row['b'], \
                         'm': row['m'], \
                         'c': row['c'], \
                         'hw': row['hw'], \
                         'rs': row['rs'], \
                         'stride': row['stride'], \
                         'padding': row['padding'], \
                         'precM': row['precM'], \
                         'precA': row['precA'], \
                         'useTensorCore': row['useTensorCore']}
                query_type = (self.op_type[0], self.op_type[1], self.prec_type)
            elif self.workload_type == 'elementwise':
                query = {'dim': row['dim'], \
                         'op': row['op'], \
                         'prec': row['prec']}
                query_type = (self.op_type)

            elif self.workload_type == 'flashattention':
                # {"batch_size": 1, "num_heads": 32, "seq_len": 2048, "head_dim": 64, "multi_query_ratio": 1, "precM": "bf16", "precA": "bf16", "useTensorCore": true, "fusion_approx_method": "flash_v2"}, ["sdpa"]
                query = {
                    'batch': row['batch'], \
                    'n_head': row['n_head'], \
                    'seq_len': row['seq_len'], \
                    'head_dim': row['head_dim'], \
                    'multi_query_ratio': 1, \
                    'prec': 'bf16', 'precM': 'bf16', 'precA': 'bf16', 'useTensorCore': True, 'fusion_approx_method': 'flash_v2'
                }
                query_type = ('flashattention_v2')
            
            estimated = self.gee.lookup(query, query_type, verbose=False, lookup_target=target, target_freq=row['avg_freq'])

            if type(estimated) == pd.core.series.Series:
                estimated = estimated.values[0]

            self.test_df.at[i, '{}_estimate'.format(target)] = estimated