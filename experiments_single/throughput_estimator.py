import os
import sys

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import yaml

sys.path.append('..')
from gee import get_gee

from gee.optimization_utils import prec_to_precision_bits

from estimator import Estimator

class ThroughputEstimator(Estimator):
    def __init__(self, train_data_csv, test_data_csv, **kwargs):
        super().__init__(train_data_csv, test_data_csv, **kwargs)

        self.throughput_mode = kwargs['throughput_mode']
        # self.gpu_config = kwargs['gpu_config']
        
        with open(kwargs['gpu_config'], 'r') as f:
            self.gpu_config = yaml.safe_load(f)['gpu_configs'] 

        # Gee initialization
        self.gee = get_gee(gpu_yaml_path=kwargs['gpu_config'], \
                           lut_yaml_path=kwargs['lut_config'], \
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

        if self.workload_type == 'gemm':
            assert (len(lut_config['gemm']) == 1) # either cuda or tc
            for key, value in lut_config['gemm'].items():
                assert(len(value) == 1) # only one lut for this
                for kkey, vvalue in value.items():
                    self.prec_type = kkey
                    # path is vvalue
                    path = os.path.realpath(os.path.join(kwargs['lut_parent_path'], 'lut', vvalue[0]['path']))
                    assert(path == os.path.realpath(self.train_data_csv))

        if self.workload_type == 'conv2d':
            assert (len(lut_config['conv2d']) == 1) # either cuda or tc
            for key, value in lut_config['conv2d'].items():
                assert(len(value) == 1) # only one lut for this
                for kkey, vvalue in value.items():
                    self.prec_type = kkey
                    # path is vvalue
                    path = os.path.realpath(os.path.join(kwargs['lut_parent_path'], 'lut', vvalue[0]['path']))
                    assert(path == os.path.realpath(self.train_data_csv))

        elif (self.workload_type == 'softmax') or (self.workload_type == 'layernorm'):
            assert (len(lut_config[self.workload_type]) == 1) # one precision
            for key, value in lut_config[self.workload_type].items():
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

    def test(self, target):

        self.test_df['{}_estimate'.format(target)] = -1

        for i, row in self.test_df.iterrows():

            if (self.workload_type == 'gemm') or (self.workload_type == 'conv2d'):
                query = row.to_dict()
                if self.workload_type == 'gemm':
                    self.gee.gemm_estimator.predict_kernel(query, 'gemm')
                    query = pd.DataFrame([query], index=[0])
                    self.gee.gemm_estimator.analytical_model.model(query, train=False)
                elif self.workload_type == 'conv2d':
                    self.gee.conv_estimator.predict_kernel(query, 'conv2d')
                    query = pd.DataFrame([query], index=[0])
                    self.gee.conv_estimator.analytical_model.model(query, train=False)
                query = query.iloc[0].to_dict()

                query['byteM'] = prec_to_precision_bits(query['precM']) / 8
                query['byteA'] = prec_to_precision_bits(query['precA']) / 8
                query['mem_footprint'] = query['batch'] * (query['dimM'] * query['dimN'] * query['byteA'] + (query['dimM'] * query['dimK'] + query['dimN'] * query['dimK']) * query['byteM'])
                query['flop'] = query['batch'] * query['dimM'] * query['dimN'] * query['dimK'] * 2
                query['comp_intensity'] = query['flop'] / query['mem_footprint']

                roofline_ridge = self.gee.gemm_estimator.gpu_roofline_ridge if self.workload_type == 'gemm' else self.gee.conv_estimator.gpu_roofline_ridge
                query['roofline_ridge'] = roofline_ridge['{}_{}'.format(row['precM'], row['useTensorCore'])]
                query['bound'] = 'flops' if (query['comp_intensity'] > query['roofline_ridge']) else 'mem'
            
            elif (self.workload_type == 'softmax') or (self.workload_type == 'layernorm'):
                query = row.to_dict()
                self.gee.nonlinear_estimator.predict_kernel(query, self.workload_type)
                query = pd.DataFrame([query], index=[0])
                self.gee.nonlinear_estimator.analytical_model.model(query, self.workload_type, train=False)
                query = query.iloc[0].to_dict()

                query['byte'] = prec_to_precision_bits(query['prec']) / 8
                query['mem_footprint'] = query['batch'] * query['dim'] * query['byte'] * 2

            elif (self.workload_type == 'elementwise'):
                query = row.to_dict()
                self.gee.elementwise_estimator.predict_kernel(query)
                query = pd.DataFrame([query], index=[0])
                self.gee.elementwise_estimator.analytical_model.model(query, train=False)
                query = query.iloc[0].to_dict()

                query['mem_footprint'] = query['dim'] * (query['num_input_arg'] * query['bytes_per_precIn'] + query['bytes_per_precOut'])

            elif (self.workload_type == 'flashattention'):
                query = {
                    'batch': row['batch'], \
                    'n_head': row['n_head'], \
                    'seq_len': row['seq_len'], \
                    'head_dim': row['head_dim'], \
                    'multi_query_ratio': 1, \
                    'prec': 'bf16', 'precM': 'bf16', 'precA': 'bf16', 'useTensorCore': True, 'fusion_approx_method': 'flash_v2', \
                    'avg_freq': row['avg_freq']
                }
                self.gee.flashattn_estimator.predict_kernel(query, 'flashattention_v2')
                query = pd.DataFrame([query], index=[0])
                self.gee.flashattn_estimator.analytical_model.model(query, train=False)
                query = query.iloc[0].to_dict()

            # throughput_mode: naive roofline
            # - bound: mem --> mem_footprint / dram_bw
            # - bound: flops --> flop / math_bw
            if self.throughput_mode == 'naive_roofline':

                if (self.workload_type == 'gemm') or (self.workload_type == 'conv2d'):
                    math_bw = self.gpu_config['{}_{}_flops'.format('cuda' if row['useTensorCore'] == False else 'tc', row['precM'])]
                    # adjust to the frequency (above is per second) --> FLOPS/SMCycle
                    math_bw = math_bw / self.gpu_config['sm_max_freq'] * query['avg_freq'] # TFLOP/s at target frequency
                    dram_bw = self.gpu_config['dram_bw'] # --> DRAM clock is fixed
                    estimated_time = query['mem_footprint'] / (dram_bw * 10 ** 6) if query['bound'] == 'mem' else query['flop'] / (math_bw * 10 ** 9) # to ms
                elif (self.workload_type == 'layernorm') or (self.workload_type == 'softmax') or (self.workload_type == 'elementwise'):
                    dram_bw = self.gpu_config['dram_bw'] # --> DRAM clock is fixed
                    estimated_time = query['mem_footprint'] / (dram_bw * 10 ** 6)

                estimated_energy = estimated_time * self.gpu_config['power_cap'] / 1000.
                self.test_df.at[i, '{}_estimate'.format(target)] = estimated_time if target == 'time' else estimated_energy

            # throughput_mode: loopnest aware roofline
            elif self.throughput_mode == 'loopnest_roofline':
                if (self.workload_type == 'gemm') or (self.workload_type == 'conv2d'):
                    estimated_time = query['cycle_throughput_max'] / (query['avg_freq'] * 10**3)
                elif (self.workload_type == 'softmax') or (self.workload_type == 'layernorm'):
                    estimated_time = max(query['t_estimated_gmem'], query['t_estimated_smem'], query['t_estimated_fp_inst'], query['t_estimated_xu_inst']) / (query['avg_freq'] * 10**3)
                elif (self.workload_type == 'elementwise'):
                    estimated_time = max(query['t_estimated_gmem'], query['t_estimated_fp']) / (query['avg_freq'] * 10**3)
                estimated_energy = estimated_time * self.gpu_config['power_cap'] / 1000.
                self.test_df.at[i, '{}_estimate'.format(target)] = estimated_time if target == 'time' else estimated_energy

            # timeline analytical
            elif self.throughput_mode == 'timeline_analytical':
                if (self.workload_type == 'gemm') or (self.workload_type == 'conv2d'):
                    estimated_time = query['t_timeline_estimated'] / (query['avg_freq'] * 10**3)
                elif (self.workload_type == 'softmax') or (self.workload_type == 'layernorm') or (self.workload_type == 'elementwise') or (self.workload_type == 'flashattention'):
                    estimated_time = query['t_estimated'] / (query['avg_freq'] * 10**3)
                estimated_energy = estimated_time * self.gpu_config['power_cap'] / 1000.
                self.test_df.at[i, '{}_estimate'.format(target)] = estimated_time if target == 'time' else estimated_energy
            
            else:
                raise NotImplementedError()