import os
import sys

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import yaml

import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split

from experiments_single.neusight_util import *

try:
    from .estimator import Estimator
except:
    from estimator import Estimator

sys.path.append('..')
from gee import get_gee
from gee.frontend_utils import * 

class NeuSightEstimator(Estimator):
    def __init__(self, gpu_config_yaml, lut_config_yaml, lut_folder_abs_path='/mnt/c/Users/KyungmiLee/Documents/gpu-energy-estimation/lut', **kwargs):
        super().__init__(gpu_config_yaml, lut_config_yaml, lut_folder_abs_path, **kwargs)

        # Train configurations
        with open(kwargs['gemm_train_config'], 'r') as f:
            self.gemm_train_config = yaml.safe_load(f)
        with open(kwargs['nonlinear_train_config'], 'r') as f:
            self.nonlinear_train_config = yaml.safe_load(f)

        # Gee initialization - use Gee's parser
        self.gee = get_gee(gpu_yaml_path=gpu_config_yaml, \
                           lut_yaml_path=lut_config_yaml, \
                           kernel_predictor_separate_prefill_decode=kwargs['kernel_predictor_separate_prefill_decode'], \
                           lut_folder_abs_path=lut_folder_abs_path, \
                           multiple_configs=kwargs['multiple_config'], \
                           gpu_configs_yaml=kwargs['multiple_gpu_configs_yaml'], \
                           dvfs_idle_power_configs_yaml=kwargs['multiple_gpu_idle_power_yaml'], \
                           dvfs_supply_voltage_configs_yaml=kwargs['multiple_gpu_supply_voltage_yaml'], \
                           no_build=True)
        
        self.dvfs_aware = kwargs['dvfs_aware']

        self.predictor = {}

        self.multiple_config = kwargs['multiple_config']
        if not self.multiple_config:
            with open(gpu_config_yaml, 'r') as f:
                self.gpu_config = yaml.safe_load(f)['gpu_configs']
        else:
            self.gpu_config = None

        with open(lut_config_yaml, 'r') as f:
            self.lut_config = yaml.safe_load(f)['lut_config']

        # Multiple config support for extrapolation (dataset collected from multiple GPUs)
        self.gpu_configs = {}
        if kwargs['multiple_config']:
            # Read the yaml file
            with open(kwargs['multiple_gpu_configs_yaml'], 'r') as f:
                multiple_gpu_configs_yaml = yaml.safe_load(f)
            
            # Load the configurations
            for key, value in multiple_gpu_configs_yaml.items():
                with open(value, 'r') as f:
                    tmp_config = yaml.safe_load(f)['gpu_configs']
                self.gpu_configs[key] = tmp_config
            
            self.multiple_config = True
        else:
            self.multiple_config = False

        if len(kwargs['pretrained_model_path']) > 0:
            with open(kwargs['pretrained_model_path'], 'r') as f:
                pretrained_model_path = yaml.safe_load(f)
            pretrained_model_path = {eval(k): v for k, v in pretrained_model_path.items()}
        else:
            pretrained_model_path = None

        # Build predictor
        if 'gemm' in self.lut_config.keys():
            for key, value in self.lut_config['gemm'].items():
                for kkey, vvalue in self.lut_config['gemm'][key].items():
                    op_type = ('gemm', key, kkey)
                    if self.dvfs_aware:
                        dfs = []
                        gpu_config_keys = []
                        pass_parsing = []
                        for entry in vvalue:
                            if entry['use_for_model']:
                                df = pd.read_csv(os.path.join(lut_folder_abs_path, entry['path']))
                                dfs.append(df)
                                if self.multiple_config:
                                    gpu_config_keys.append(entry['gpu_config_key'])
                                pass_parsing.append(entry['prepared'])
                        self.predictor[op_type] = self._build_predictor_dvfs(dfs, op_type, pretrained=kwargs['pretrained'] and (op_type in pretrained_model_path.keys()), \
                                                                             pretrained_time_model_pth=None if ((not kwargs['pretrained']) or (op_type not in pretrained_model_path.keys())) else pretrained_model_path[op_type]['time'], \
                                                                             pretrained_energy_model_pth=None if ((not kwargs['pretrained']) or (op_type not in pretrained_model_path.keys())) else pretrained_model_path[op_type]['power'], \
                                                                             gpu_config_keys=gpu_config_keys, pass_parsing=pass_parsing)
                        if kwargs['save_model']:
                            self._save_predictor(self.predictor[op_type], op_type, kwargs['save_to'])
                    else:
                        for entry in vvalue:
                            if entry['main']:
                                df = pd.read_csv(os.path.join(lut_folder_abs_path, entry['path']))
                                self.predictor[op_type] = self._build_predictor(df, op_type, pretrained=kwargs['pretrained'] and (op_type in pretrained_model_path.keys()), \
                                                                                pretrained_time_model_pth=None if ((not kwargs['pretrained']) or (op_type not in pretrained_model_path.keys())) else pretrained_model_path[op_type]['time'], \
                                                                                pretrained_energy_model_pth=None if ((not kwargs['pretrained']) or (op_type not in pretrained_model_path.keys())) else pretrained_model_path[op_type]['power'], \
                                                                                pass_parsing=entry['prepared'])
                                if kwargs['save_model']:
                                    self._save_predictor(self.predictor[op_type], op_type, kwargs['save_to'])
                                break

        if 'conv2d' in self.lut_config.keys():
            for key, value in self.lut_config['conv2d'].items():
                for kkey, vvalue in self.lut_config['conv2d'][key].items():
                    op_type = ('conv2d', key, kkey)
                    if self.dvfs_aware:
                        dfs = []
                        gpu_config_keys = []
                        pass_parsing = []
                        for entry in vvalue:
                            if entry['use_for_model']:
                                df = pd.read_csv(os.path.join(lut_folder_abs_path, entry['path']))
                                dfs.append(df)
                                if self.multiple_config:
                                    gpu_config_keys.append(entry['gpu_config_key'])
                                pass_parsing.append(entry['prepared'])
                        self.predictor[op_type] = self._build_predictor_dvfs(dfs, op_type, pretrained=kwargs['pretrained'] and (op_type in pretrained_model_path.keys()), \
                                                                             pretrained_time_model_pth=None if ((not kwargs['pretrained']) or (op_type not in pretrained_model_path.keys())) else pretrained_model_path[op_type]['time'], \
                                                                             pretrained_energy_model_pth=None if ((not kwargs['pretrained']) or (op_type not in pretrained_model_path.keys())) else pretrained_model_path[op_type]['power'], \
                                                                             gpu_config_keys=gpu_config_keys, pass_parsing=pass_parsing)
                        if kwargs['save_model']:
                            self._save_predictor(self.predictor[op_type], op_type, kwargs['save_to'])
                    else:
                        for entry in vvalue:
                            if entry['main']:
                                df = pd.read_csv(os.path.join(lut_folder_abs_path, entry['path']))
                                self.predictor[op_type] = self._build_predictor(df, op_type, pretrained=kwargs['pretrained'] and (op_type in pretrained_model_path.keys()), \
                                                                                pretrained_time_model_pth=None if ((not kwargs['pretrained']) or (op_type not in pretrained_model_path.keys())) else pretrained_model_path[op_type]['time'], \
                                                                                pretrained_energy_model_pth=None if ((not kwargs['pretrained']) or (op_type not in pretrained_model_path.keys())) else pretrained_model_path[op_type]['power'])
                                if kwargs['save_model']:
                                    self._save_predictor(self.predictor[op_type], op_type, kwargs['save_to'])
                                break
                    
        if 'softmax' in self.lut_config.keys():
            for key, value in self.lut_config['softmax'].items():
                op_type = ('softmax', key)
                if self.dvfs_aware:
                    dfs = []
                    gpu_config_keys = []
                    pass_parsing = []
                    for entry in value:
                        if entry['use_for_model']:
                            df = pd.read_csv(os.path.join(lut_folder_abs_path, entry['path']))
                            dfs.append(df)
                            if self.multiple_config:
                                gpu_config_keys.append(entry['gpu_config_key'])
                            pass_parsing.append(entry['prepared'])
                    self.predictor[op_type] = self._build_predictor_dvfs(dfs, op_type, pretrained=kwargs['pretrained'] and (op_type in pretrained_model_path.keys()), \
                                                                         pretrained_time_model_pth=None if ((not kwargs['pretrained']) or (op_type not in pretrained_model_path.keys())) else pretrained_model_path[op_type]['time'], \
                                                                         pretrained_energy_model_pth=None if ((not kwargs['pretrained']) or (op_type not in pretrained_model_path.keys())) else pretrained_model_path[op_type]['power'], \
                                                                         gpu_config_keys=gpu_config_keys, pass_parsing=pass_parsing)
                    if kwargs['save_model']:
                        self._save_predictor(self.predictor[op_type], op_type, kwargs['save_to'])
                else:
                    for entry in value:
                        if entry['main']:
                            df = pd.read_csv(os.path.join(lut_folder_abs_path, entry['path']))
                            self.predictor[op_type] = self._build_predictor(df, op_type, pretrained=kwargs['pretrained'] and (op_type in pretrained_model_path.keys()), \
                                                                            pretrained_time_model_pth=None if ((not kwargs['pretrained']) or (op_type not in pretrained_model_path.keys())) else pretrained_model_path[op_type]['time'], \
                                                                            pretrained_energy_model_pth=None if ((not kwargs['pretrained']) or (op_type not in pretrained_model_path.keys())) else pretrained_model_path[op_type]['power'])
                            if kwargs['save_model']:
                                self._save_predictor(self.predictor[op_type], op_type, kwargs['save_to'])
                            break

        if 'layernorm' in self.lut_config.keys():
            for key, value in self.lut_config['layernorm'].items():
                op_type = ('layernorm', key)
                if self.dvfs_aware:
                    dfs = []
                    gpu_config_keys = []
                    pass_parsing = []
                    for entry in value:
                        if entry['use_for_model']:
                            df = pd.read_csv(os.path.join(lut_folder_abs_path, entry['path']))
                            dfs.append(df)
                            if self.multiple_config:
                                gpu_config_keys.append(entry['gpu_config_key'])
                            pass_parsing.append(entry['prepared'])
                    self.predictor[op_type] = self._build_predictor_dvfs(dfs, op_type, pretrained=kwargs['pretrained'] and (op_type in pretrained_model_path.keys()), \
                                                                         pretrained_time_model_pth=None if ((not kwargs['pretrained']) or (op_type not in pretrained_model_path.keys())) else pretrained_model_path[op_type]['time'], \
                                                                         pretrained_energy_model_pth=None if ((not kwargs['pretrained']) or (op_type not in pretrained_model_path.keys())) else pretrained_model_path[op_type]['power'], \
                                                                         gpu_config_keys=gpu_config_keys, pass_parsing=pass_parsing)
                    if kwargs['save_model']:
                        self._save_predictor(self.predictor[op_type], op_type, kwargs['save_to'])
                else:
                    for entry in value:
                        if entry['main']:
                            df = pd.read_csv(os.path.join(lut_folder_abs_path, entry['path']))
                            self.predictor[op_type] = self._build_predictor(df, op_type, pretrained=kwargs['pretrained'] and (op_type in pretrained_model_path.keys()), \
                                                                            pretrained_time_model_pth=None if ((not kwargs['pretrained']) or (op_type not in pretrained_model_path.keys())) else pretrained_model_path[op_type]['time'], \
                                                                            pretrained_energy_model_pth=None if ((not kwargs['pretrained']) or (op_type not in pretrained_model_path.keys())) else pretrained_model_path[op_type]['power'])
                            if kwargs['save_model']:
                                self._save_predictor(self.predictor[op_type], op_type, kwargs['save_to'])
                            break

        if 'elementwise' in self.lut_config.keys():
            op_type = ('elementwise',)
            if self.dvfs_aware:
                dfs = []
                gpu_config_keys = []
                pass_parsing = []
                for entry in self.lut_config['elementwise']:
                    if entry['use_for_model']:
                        df = pd.read_csv(os.path.join(lut_folder_abs_path, entry['path']))
                        dfs.append(df)
                        if self.multiple_config:
                            gpu_config_keys.append(entry['gpu_config_key'])
                        pass_parsing.append(entry['prepared'])
                self.predictor[op_type] = self._build_predictor_dvfs(dfs, op_type, pretrained=kwargs['pretrained'] and (op_type in pretrained_model_path.keys()), \
                                                                     pretrained_time_model_pth=None if ((not kwargs['pretrained']) or (op_type not in pretrained_model_path.keys())) else pretrained_model_path[op_type]['time'], \
                                                                     pretrained_energy_model_pth=None if ((not kwargs['pretrained']) or (op_type not in pretrained_model_path.keys()))else pretrained_model_path[op_type]['power'], \
                                                                     gpu_config_keys=gpu_config_keys, pass_parsing=pass_parsing)
                if kwargs['save_model']:
                    self._save_predictor(self.predictor[op_type], op_type, kwargs['save_to'])
            else:
                for entry in self.lut_config['elementwise']:
                    if entry['main']:
                        df = pd.read_csv(os.path.join(lut_folder_abs_path, entry['path']))
                        self.predictor[op_type] = self._build_predictor(df, op_type, pretrained=kwargs['pretrained'] and (op_type in pretrained_model_path.keys()), \
                                                                        pretrained_time_model_pth=None if ((not kwargs['pretrained']) or (op_type not in pretrained_model_path.keys())) else pretrained_model_path[op_type]['time'], \
                                                                        pretrained_energy_model_pth=None if ((not kwargs['pretrained']) or (op_type not in pretrained_model_path.keys())) else pretrained_model_path[op_type]['power'])
                        if kwargs['save_model']:
                            self._save_predictor(self.predictor[op_type], op_type, kwargs['save_to'])
                        break

    def _save_predictor(self, predictor, op_type, path):
        time_model = predictor[1]
        power_model = predictor[2]
        op_type_str = '_'.join(map(str, op_type))

        torch.save(time_model.state_dict(), os.path.join(path, '{}_time.pth'.format(op_type_str)))
        torch.save(power_model.state_dict(), os.path.join(path, '{}_power.pth'.format(op_type_str)))
    
    def _preprocess_df(self, df, op_type, train=True, verbose=False, target_gpu_config=None, \
                       gpu_max_freq=-1, pass_parsing=False):
        if (op_type[0] == 'gemm') or (op_type[0] == 'conv2d'):
            config_key = op_type[1] + '_' + op_type[2].split('_')[0] + '_flops'
            gpu_flops = self.gpu_config[config_key] if target_gpu_config is None else target_gpu_config[config_key]
        elif op_type[0] == 'softmax':
            config_key = 'cuda_fp32_flops'
            gpu_flops = self.gpu_config[config_key] if target_gpu_config is None else target_gpu_config[config_key]
            # config_key = 'xu_bw_per_sm'
            # gpu_flops = (self.gpu_config[config_key] * self.gpu_config['num_sm']) * self.gpu_config['sm_max_freq'] / 10**6 # TFLOPs
        elif op_type[0] == 'layernorm':
            config_key = 'cuda_fp32_flops'
            gpu_flops = self.gpu_config[config_key] if target_gpu_config is None else target_gpu_config[config_key]
        elif op_type[0] == 'elementwise':
            config_key = 'cuda_fp32_flops'
            gpu_flops = self.gpu_config[config_key] if target_gpu_config is None else target_gpu_config[config_key]

        gpu_sm = self.gpu_config['num_sm'] if target_gpu_config is None else target_gpu_config['num_sm']
        gpu_hbm_bw = self.gpu_config['dram_bw'] if target_gpu_config is None else target_gpu_config['dram_bw']
        gpu_hbm = self.gpu_config['dram_size'] if target_gpu_config is None else target_gpu_config['dram_size']
        gpu_l2 = self.gpu_config['l2_size'] if target_gpu_config is None else target_gpu_config['l2_size']

        if gpu_max_freq < 0:
            gpu_max_freq = self.gpu_config['sm_max_freq'] if target_gpu_config is None else target_gpu_config['sm_max_freq']

        roofline_max_freq = self.gpu_config['sm_max_freq'] if target_gpu_config is None else target_gpu_config['sm_max_freq']
        df['gpu_flops'] = gpu_flops * df['avg_freq'] / roofline_max_freq

        if (op_type[0] == 'gemm') or (op_type[0] == 'conv2d'):
            if train:
                if not pass_parsing:
                    if op_type[0] == 'gemm':
                        self.gee.gemm_estimator.kernel_parser.parse_dataframe(df, 'gemm')
                    elif op_type[0] == 'conv2d':
                        self.gee.conv_estimator.kernel_parser.parse_dataframe(df, 'conv2d')
            else:
                assert (len(df) == 1)
                df = dict(df.iloc[0])
                if op_type[0] == 'gemm':
                    self.gee.gemm_estimator.predict_kernel(df, 'gemm')
                elif op_type[0] == 'conv2d':
                    self.gee.conv_estimator.predict_kernel(df, 'conv2d')
                df = pd.DataFrame([df], index=[0])

                # if verbose:
                #     print(df.to_markdown())
        
            df['b_tile'] = 1
            df['m_tile'] = df.apply(lambda row: min(row['block_tile_M'], row['dimM']), axis=1) # min(df['block_tile_M'], df['dimM'])
            df['n_tile'] = df.apply(lambda row: min(row['block_tile_N'], row['dimN']), axis=1)
            df['k_tile'] = df['totalK']

            df['_splitK_batch'] = df['splitK_batch'].apply(lambda x: max(x, 1))

            byteM = prec_to_precision_bits(df['precM'].iloc[0]) / 8
            byteA = prec_to_precision_bits(df['precA'].iloc[0]) / 8

            df['flop_per_tile'] = df['b_tile'] * df['m_tile'] * df['n_tile'] * df['k_tile'] * 2 / 10**9 # to gflops
            df['mem_per_tile'] = df['b_tile'] * (df['m_tile'] * df['n_tile'] * byteA + df['m_tile'] * df['k_tile'] * byteM + df['n_tile'] * df['k_tile'] * byteM) / 2 ** 20 # in MB
            df['total_flop'] = df['batch'] * df['dimM'] * df['dimN'] * df['dimK'] * 2 / 10**9 # gflops
            df['total_mem'] = df['batch'] * df['_splitK_batch'] * (df['dimM'] * df['dimN'] * byteA + df['dimM'] * df['totalK'] * byteM + df['dimN'] * df['totalK'] * byteM) / 2**30 # GB

        elif op_type[0] == 'softmax':
            if train:
                self.gee.nonlinear_estimator.kernel_parser.parse_dataframe(df, 'softmax')
            else:
                assert (len(df) == 1)
                df = dict(df.iloc[0])
                self.gee.nonlinear_estimator.predict_kernel(df, 'softmax')
                df = pd.DataFrame([df], index=[0])

            df['b_tile'] = df.apply(lambda row: min(row['block_tile_batch'], row['batch']), axis=1)
            df['d_tile'] = df.apply(lambda row: min(row['block_tile_softmax'], row['dim']), axis=1)

            byteD = prec_to_precision_bits(df['prec'].iloc[0]) / 8

            df['flop_per_tile'] = df['b_tile'] * df['d_tile'] * 5 / 10**9
            df['mem_per_tile'] = (df['b_tile'] * df['d_tile']) * 2 * byteD / 2 ** 20
            df['total_flop'] = df['batch'] * df['dim'] * 5 / 10**9
            df['total_mem'] = df['batch'] * df['dim'] * 2 * byteD / 2 ** 30

        elif op_type[0] == 'layernorm':
            if train:
                self.gee.nonlinear_estimator.kernel_parser.parse_dataframe(df, 'layernorm')
            else:
                assert (len(df) == 1)
                df = dict(df.iloc[0])
                self.gee.nonlinear_estimator.predict_kernel(df, 'layernorm')
                df = pd.DataFrame([df], index=[0])

            df['b_tile'] = df.apply(lambda row: min(row['block_tile_batch'], row['batch']), axis=1)
            df['d_tile'] = df.apply(lambda row: min(row['block_tile_layernorm'], row['dim']), axis=1)

            byteD = prec_to_precision_bits(df['prec'].iloc[0]) / 8

            df['flop_per_tile'] = df['b_tile'] * df['d_tile'] * 6 / 10**9
            df['mem_per_tile'] = (df['b_tile'] * df['d_tile']) * 2 * byteD / 2 ** 20
            df['total_flop'] = df['batch'] * df['dim'] * 6 / 10**9
            df['total_mem'] = df['batch'] * df['dim'] * 2 * byteD / 2 ** 30

        elif op_type[0] == 'elementwise':
            if train:
                self.gee.elementwise_estimator.kernel_parser.parse_dataframe(df)
            else:
                df = dict(df.iloc[0])
                self.gee.elementwise_estimator.predict_kernel(df)
                df = pd.DataFrame([df], index=[0])

            df['d_tile'] = df.apply(lambda row: min(row['block_tile'], row['dim']), axis=1)

            def get_datatype_bytes(row, inout='in'):
                op = row['op']
                prec = row['prec']

                if inout == 'in':
                    return prec_to_precision_bits(prec) / 8
                else:
                    if op == 'typecast_to_fp32':
                        return 4
                    elif op == 'typecast_to_bf16':
                        return 2
                    else:
                        return prec_to_precision_bits(prec) / 8
                    
            def get_num_input_args(op):
                if op in ['pointwise_mul', 'pointwise_add', 'unspecified_tensor']:
                    return 2
                else:
                    return 1

            df['byteIn'] = df.apply(lambda row: get_datatype_bytes(row, 'in'), axis=1)
            df['byteOut'] = df.apply(lambda row: get_datatype_bytes(row, 'out'), axis=1)
            df['num_input_args'] = df['op'].apply(lambda x: get_num_input_args(x))

            df['flop_per_tile'] = df['d_tile'] / 10**9
            df['mem_per_tile'] = (df['d_tile'] * df['num_input_args'] * df['byteIn'] + df['d_tile'] * df['byteOut']) / 2**20
            df['total_flop'] = df['dim'] / 10**9
            df['total_mem'] = (df['dim'] * df['num_input_args'] * df['byteIn'] + df['dim'] * df['byteOut']) / 2**30

        df['comp_intensity'] = df['total_flop'] / df['total_mem'] # gflops/GB --> flops/B
        # df['roofline_bw'] = df['comp_intensity'].apply(lambda x : min(gpu_flops * 1000., x * gpu_hbm_bw)) # gflops
        df['roofline_bw'] = df.apply(lambda row: min(row['gpu_flops'] * 1000., row['comp_intensity'] * gpu_hbm_bw), axis=1)

        df['total_tiles'] = df['total_block_tiles'] # Ignoring split-K kernels # df['tile_info'].apply(lambda x : x[0] * x[1] * x[2])
        df['num_waves'] = np.ceil(df['total_tiles'] / gpu_sm)

        df['mem_dram'] = df['mem_per_tile'] * df['num_waves']
        df['mem_l2'] = df['mem_per_tile'] * df['num_waves']

        df['flop_per_wave'] = df['flop_per_tile'] * gpu_sm

        # tflops/sm, gb/s/sm, mb/sm, kb/sm
        norm_vector = [gpu_flops / gpu_sm, gpu_hbm_bw / gpu_sm , gpu_hbm * 1024 / gpu_sm, gpu_l2 * 1024 / gpu_sm]

        if self.dvfs_aware:
            norm_vector.append(gpu_max_freq)
        
        if train:
            df['power'] = df['energy'] / (df['time'] / 1000.)

            # time = num_wave * flop_per_wave / ebw
            # ebw = roofline_bw * bw_util
            # time = num_wave * flop_per_wave / (roofline_bw * bw_util)
            # bw_util = num_wave * flop_per_wave / (roofline_bw * time)
            df['bw_util'] = df['num_waves']  * df['flop_per_wave'] / df['roofline_bw'] / (df['time'] / 1000)
            return norm_vector
        else:
            return df, norm_vector

    def _train_model(self, df, norm_vector, train_config, pretrained=False, pretrained_time_model_pth=None, pretrained_energy_model_pth=None, \
                     gpu_config_keys=[]):
        
        dataset = NeuSightDataset(df, norm_vector, train_config['log_normalize_inputs'])
        
        train_dataloader = DataLoader(dataset, batch_size=train_config['batch_size'], shuffle=True)

        def MAPELoss(pred, target):
            return torch.mean(torch.abs((target - pred) / target))
        def SMAPELoss(pred, target):
            return torch.mean(torch.abs((target - pred) / (target + pred)))
        
        if train_config['loss'] == 'MAPE':
            criterion = MAPELoss
        elif train_config['loss'] == 'SMAPE':
            criterion = SMAPELoss
        else:
            raise TypeError("This loss of type {} is not supported!".format(train_config['loss']))
    
        time_model = NeuSightMLP(n_layer=train_config['MLP_layers_time'], hidden_dim=train_config['MLP_hidden_time'], input_dim=5 if self.dvfs_aware else 4, output_dim=2)
        time_optim = torch.optim.AdamW(time_model.parameters(), lr=eval(train_config['lr_time']))

        if pretrained:
            time_model.load_state_dict(torch.load(pretrained_time_model_pth))
        else:
            loss, val_error = train_time_model(time_model, train_dataloader, train_dataloader, time_optim, criterion, train_config['epoch_time'])
            print("Last epoch validtion error: {:.2f} %".format(val_error[-1] * 100))

        power_model = NeuSightMLP(n_layer=train_config['MLP_layers_power'], hidden_dim=train_config['MLP_hidden_power'], input_dim=5 if self.dvfs_aware else 4, output_dim=train_config['MLP_odim_power'])    
        power_optim = torch.optim.AdamW(power_model.parameters(), lr=eval(train_config['lr_power']))

        if 'max_power_across_all_gpus' in train_config.keys():
            self.train_power_cap = train_config['max_power_across_all_gpus']
            self.test_power_cap = train_config['max_power_across_all_gpus']
        else:
            if self.multiple_config:
                self.train_power_cap = -1
            else:
                self.train_power_cap = self.gpu_config['power_cap']
            self.test_power_cap = -1

        if pretrained:
            power_model.load_state_dict(torch.load(pretrained_energy_model_pth))
        else:
            loss, val_error = train_power_model(time_model, power_model, train_dataloader, train_dataloader, \
                                                power_optim, criterion, train_config['epoch_power'], self.train_power_cap, \
                                                direct_power=train_config['train_power_model_using_true_time'])
            print("Last epoch validtion error: {:.2f} %".format(val_error[-1] * 100))
            
        return time_model, power_model

    def _build_predictor(self, df, op_type, pretrained=False, pretrained_time_model_pth=None, pretrained_energy_model_pth=None, pass_parsing=False):
        norm_vector = self._preprocess_df(df, op_type, pass_parsing=pass_parsing)
        time_model, power_model = self._train_model(df, norm_vector, self.gemm_train_config if (op_type[0] == 'gemm' or op_type[0] == 'conv2d') else self.nonlinear_train_config, pretrained, pretrained_time_model_pth, pretrained_energy_model_pth)

        self.gpu_max_freq = -1
        return (norm_vector, time_model, power_model)
    
    def _build_predictor_dvfs(self, dfs, op_type, pretrained=False, pretrained_time_model_pth=None, pretrained_energy_model_pth=None, \
                              gpu_config_keys=[], pass_parsing=[]):
        self.gpu_max_freq = -1
        train_config = self.gemm_train_config if (op_type[0] == 'gemm' or op_type[0] == 'conv2d') else self.nonlinear_train_config
        if 'max_freq_across_all_gpus' in train_config.keys():
            self.gpu_max_freq = train_config['max_freq_across_all_gpus']
        if self.multiple_config:
            norm_vector = []
            for i, df in enumerate(dfs):
                _norm_vector = self._preprocess_df(df, op_type, target_gpu_config=self.gpu_configs[gpu_config_keys[i]], \
                                                   gpu_max_freq=self.gpu_max_freq, pass_parsing=pass_parsing[i])
                norm_vector.append(_norm_vector)
        else:
            for df in dfs:
                norm_vector = self._preprocess_df(df, op_type, gpu_max_freq=self.gpu_max_freq, pass_parsing=pass_parsing[i])
        time_model, power_model = self._train_model(dfs, norm_vector, train_config, pretrained, pretrained_time_model_pth, pretrained_energy_model_pth, \
                                                    gpu_config_keys=gpu_config_keys)

        return (norm_vector, time_model, power_model)
    
    def predict_single_kernel(self, query, op_type, **kwargs):
        target_gpu_config = kwargs['target_gpu_config']

        query['avg_freq'] = kwargs['target_freq']

        _, time_model, power_model = self.predictor[op_type]

        config = self.gemm_train_config if (op_type[0] == 'gemm' or op_type[0] == 'conv2d') else self.nonlinear_train_config

        query = pd.DataFrame([query], index=[0])
        query, norm_vector = self._preprocess_df(query, op_type, train=False, verbose=kwargs['verbose'] if 'verbose' in kwargs.keys() else False, \
                                                 target_gpu_config=target_gpu_config, gpu_max_freq=self.gpu_max_freq)
        query = query.iloc[0].to_dict()

        if kwargs['verbose']:
            for k, v in query.items():
                print(k, v)
        
        if self.dvfs_aware:
            input_feature = np.asarray([query['flop_per_tile'], query['mem_per_tile'], query['mem_dram'], query['mem_l2'], kwargs['target_freq']])
        else:
            input_feature = np.asarray([query['flop_per_tile'], query['mem_per_tile'], query['mem_dram'], query['mem_l2']])
        if input_feature.ndim < 2:
            input_feature = np.expand_dims(input_feature, axis=0)
        input_feature = input_feature / np.asarray(norm_vector)

        if config['log_normalize_inputs']:
            input_feature[:, :4] = np.log2(input_feature[:, :4])

        input_feature = torch.Tensor(input_feature)
        
        time_pred = time_model(input_feature.float())
        time_alpha = torch.sigmoid(time_pred[:, 0]).item()
        time_beta = torch.sigmoid(time_pred[:, 1]).item()

        wave = query['num_waves']
        roofline = query['roofline_bw']
        flop_per_wave = query['flop_per_wave']
        time_util = time_alpha - time_beta / wave
        ebw = time_util * roofline
        time_estimated = flop_per_wave / ebw * wave * 1000.

        power_util = torch.sigmoid(power_model(input_feature.float())).item()

        # ep = power_util * self.train_power_cap
        ep = (power_util * self.test_power_cap) if self.test_power_cap > 0 else (power_util * target_gpu_config['power_cap'] if target_gpu_config is not None else power_util * self.gpu_config['power_cap'])
        energy_estimated = time_estimated * ep / 1000. # ms --> s

        return (time_estimated, energy_estimated)
    
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
            estimated = self.predict_single_kernel(resultvars, query_type, \
                                                   verbose=verbose, target_freq=target_freq, \
                                                   target_gpu_config=None)
            
            if lookup_target == 'time':
                energy_list.append(estimated[0])
            elif lookup_target == 'energy':
                energy_list.append(estimated[1])
            else:
                energy_list.append((estimated[0], -1, estimated[1]))
            
        return energy_list