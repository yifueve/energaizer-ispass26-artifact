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

from neusight_util import *

from estimator import Estimator

sys.path.append('..')
from gee import get_gee

class NeuSightEstimator(Estimator):
    def __init__(self, train_data_csv, test_data_csv, **kwargs):
        super().__init__(train_data_csv, test_data_csv, **kwargs)

        self.train_df = pd.read_csv(train_data_csv)

        # Gee initialization
        self.gee = get_gee(gpu_yaml_path=kwargs['gpu_config'], \
                           lut_yaml_path=kwargs['lut_config'], \
                           lut_folder_abs_path=os.path.realpath(os.path.join(kwargs['lut_parent_path'], 'lut')))

        with open(kwargs['gpu_config']) as f:
            try:
                self.gpu_config = yaml.safe_load(f)['gpu_configs']
            except yaml.YAMLError as exc:
                print(exc)
                print("Error while loading gpu configuration yaml file")
                exit()

        self.workload_type = kwargs['workload_type']
        
        # gpu information needed: #SM, FLOPs, HBM BW, HBM Size, L2 Size, Power Cap

        self.train_config_set = False

        self.preprocess()
        
    def preprocess(self):
        self.reset_testdf()

        if (self.workload_type == 'gemm') or (self.workload_type == 'conv2d'):
            # which flop number to use? --> check train_df
            precM = self.train_df['precM'].iloc[0]
            precA = self.train_df['precA'].iloc[0]
            useTensorCore = self.train_df['useTensorCore'].iloc[0]

            # assert that all elements in train_df and test_df should have the same precision + tensorcore option
            precMs = self.train_df['precM'].unique()
            assert ((len(precMs) == 1) and precMs[0] == precM)
            precAs = self.train_df['precA'].unique()
            assert ((len(precAs) == 1) and precAs[0] == precA)
            tcs = self.train_df['useTensorCore'].unique()
            assert ((len(tcs) == 1) and tcs[0] == useTensorCore)

            precMs = self.test_df['precM'].unique()
            assert ((len(precMs) == 1) and precMs[0] == precM)
            precAs = self.test_df['precA'].unique()
            assert ((len(precAs) == 1) and precAs[0] == precA)
            tcs = self.test_df['useTensorCore'].unique()
            assert ((len(tcs) == 1) and tcs[0] == useTensorCore)

        elif (self.workload_type == 'softmax') or (self.workload_type == 'layernorm'):
            prec = self.train_df['prec'].iloc[0]

            precs = self.train_df['prec'].unique()
            assert((len(precs) == 1) and precs[0] == prec)

            precs = self.test_df['prec'].unique()
            assert((len(precs) == 1) and precs[0] == prec)

        # get correct gpu configuration
        if (self.workload_type == 'gemm') or (self.workload_type == 'conv2d'):
            config_key = ('tc' if useTensorCore else 'cuda') + '_' + precM + '_flops'
            gpu_flops = self.gpu_config[config_key]
        elif self.workload_type == 'softmax':
            config_key = 'cuda_fp32_flops'
            gpu_flops = self.gpu_config[config_key]
            # gpu_flops = (self.gpu_config[config_key] * self.gpu_config['num_sm']) * self.gpu_config['sm_max_freq'] / 10**6 # TFLOPs
        elif self.workload_type == 'layernorm':
            config_key = 'cuda_fp32_flops'
            gpu_flops = self.gpu_config[config_key]
        elif self.workload_type == 'elementwise':
            config_key = 'cuda_fp32_flops'
            gpu_flops = self.gpu_config[config_key]

        gpu_sm = self.gpu_config['num_sm']
        gpu_hbm_bw = self.gpu_config['dram_bw']
        gpu_hbm = self.gpu_config['dram_size']
        gpu_l2 = self.gpu_config['l2_size']

        self.train_norm_vector = self.preprocess_df(self.train_df, gpu_sm, gpu_flops, gpu_hbm_bw, gpu_hbm, gpu_l2)
        self.test_norm_vector = self.preprocess_df(self.test_df, gpu_sm, gpu_flops, gpu_hbm_bw, gpu_hbm, gpu_l2)
        # Actually these two norm vectors are the same - not data dependent

    def preprocess_df(self, df, gpu_sm, gpu_flops, gpu_hbm_bw, gpu_hbm, gpu_l2):
        # run gee preprocessing - a bit redundant for train_df though...

        if (self.workload_type == 'gemm') or (self.workload_type == 'conv2d'):

            if self.workload_type == 'gemm':
                self.gee.gemm_estimator.kernel_parser.parse_dataframe(df, 'gemm')
            elif self.workload_type == 'conv2d':
                self.gee.conv_estimator.kernel_parser.parse_dataframe(df, 'conv2d')
        
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

        elif self.workload_type == 'softmax':
            self.gee.nonlinear_estimator.kernel_parser.parse_dataframe(df, 'softmax')

            df['b_tile'] = df.apply(lambda row: min(row['block_tile_batch'], row['batch']), axis=1)
            df['d_tile'] = df.apply(lambda row: min(row['block_tile_softmax'], row['dim']), axis=1)

            byteD = prec_to_precision_bits(df['prec'].iloc[0]) / 8

            df['flop_per_tile'] = df['b_tile'] * df['d_tile'] * 5 / 10**9
            df['mem_per_tile'] = (df['b_tile'] * df['d_tile']) * 2 * byteD / 2 ** 20
            df['total_flop'] = df['batch'] * df['dim'] * 5 / 10**9
            df['total_mem'] = df['batch'] * df['dim'] * 2 * byteD / 2 ** 30

        elif self.workload_type == 'layernorm':
            self.gee.nonlinear_estimator.kernel_parser.parse_dataframe(df, 'layernorm')

            df['b_tile'] = df.apply(lambda row: min(row['block_tile_batch'], row['batch']), axis=1)
            df['d_tile'] = df.apply(lambda row: min(row['block_tile_layernorm'], row['dim']), axis=1)

            byteD = prec_to_precision_bits(df['prec'].iloc[0]) / 8

            df['flop_per_tile'] = df['b_tile'] * df['d_tile'] * 6 / 10**9
            df['mem_per_tile'] = (df['b_tile'] * df['d_tile']) * 2 * byteD / 2 ** 20
            df['total_flop'] = df['batch'] * df['dim'] * 6 / 10**9
            df['total_mem'] = df['batch'] * df['dim'] * 2 * byteD / 2 ** 30

        elif self.workload_type == 'elementwise':
            self.gee.elementwise_estimator.kernel_parser.parse_dataframe(df)
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
        df['roofline_bw'] = df['comp_intensity'].apply(lambda x : min(gpu_flops * 1000., x * gpu_hbm_bw)) # gflops

        df['total_tiles'] = df['total_block_tiles'] # Ignoring split-K kernels # df['tile_info'].apply(lambda x : x[0] * x[1] * x[2])
        df['num_waves'] = np.ceil(df['total_tiles'] / gpu_sm)

        df['mem_dram'] = df['mem_per_tile'] * df['num_waves']
        df['mem_l2'] = df['mem_per_tile'] * df['num_waves']

        df['flop_per_wave'] = df['flop_per_tile'] * gpu_sm
        df['power'] = df['energy'] / (df['time'] / 1000.)

        # tflops/sm, gb/s/sm, mb/sm, kb/sm
        norm_vector = [gpu_flops / gpu_sm, gpu_hbm_bw / gpu_sm , gpu_hbm * 1024 / gpu_sm, gpu_l2 * 1024 / gpu_sm]

        # time = num_wave * flop_per_wave / ebw
        # ebw = roofline_bw * bw_util
        # time = num_wave * flop_per_wave / (roofline_bw * bw_util)
        # bw_util = num_wave * flop_per_wave / (roofline_bw * time)
        df['bw_util'] = df['num_waves']  * df['flop_per_wave'] / df['roofline_bw'] / (df['time'] / 1000)

        return norm_vector

    def set_train_config(self, **kwargs):
        self.train_config_set = True

        self.log_normalize_inputs = kwargs['log_normalize_inputs']
        self.validation_fraction = kwargs['validation_fraction']
        self.batch_size = kwargs['batch_size']
        self.loss = kwargs['loss']
        
        self.MLP_layers_time = kwargs['MLP_layers_time']
        self.MLP_hidden_time = kwargs['MLP_hidden_time']
        self.MLP_layers_power = kwargs['MLP_layers_power']
        self.MLP_hidden_power = kwargs['MLP_hidden_power']

        self.lr_time = eval(kwargs['lr_time'])
        self.lr_power = eval(kwargs['lr_power'])
        self.epoch_time = kwargs['epoch_time']
        self.epoch_power = kwargs['epoch_power']
        
        self.MLP_odim_power = kwargs['MLP_odim_power']
        assert((self.MLP_odim_power == 1))

        self.train_power_model_using_true_time = kwargs['train_power_model_using_true_time']

    def set_train_config_from_yaml(self, path):
        with open(path) as f:
            try:
                config = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(exc)
                print("Error while loading gpu configuration yaml file")
                exit()
        
        self.set_train_config(**config)

    def train(self, target, plot_loss=True, retrain=False):
        assert (self.train_config_set == True, 'Train configuration should have been declared before. Use set_train_config function.')
        if (self.trained[target] and not retrain):
            print("Target {} already trained. Skipping training.".format(target))
            return

        dataset = NeuSightDataset(self.train_df, self.train_norm_vector, self.log_normalize_inputs)
        if self.validation_fraction > 0:
            assert (self.validation_fraction < 1, 'Validation set size has to be defined between 0~1, indicating the fraction.')
            train_size = int((1 - self.validation_fraction) * len(dataset))
            val_size = len(dataset) - train_size
            train, val = random_split(dataset, (train_size, val_size))
        else:
            print("Validation set is not used. Using test dataset as validation set.")
            test_dataset = NeuSightDataset(self.test_df, self.test_norm_vector, self.log_normalize_inputs)
            train = dataset
            val = test_dataset
        
        train_dataloader = DataLoader(train, batch_size=self.batch_size, shuffle=True)
        val_dataloader = DataLoader(val, batch_size=self.batch_size, shuffle=False)

        def MAPELoss(pred, target):
            return torch.mean(torch.abs((target - pred) / target))
        def SMAPELoss(pred, target):
            return torch.mean(torch.abs((target - pred) / (target + pred)))
        
        if self.loss == 'MAPE':
            criterion = MAPELoss
        elif self.loss == 'SMAPE':
            criterion = SMAPELoss
        else:
            raise TypeError("This loss of type {} is not supported!".format(self.loss))
    
        if target == 'time':
            time_model = NeuSightMLP(n_layer=self.MLP_layers_time, hidden_dim=self.MLP_hidden_time, input_dim=4, output_dim=2)
            time_optim = torch.optim.AdamW(time_model.parameters(), lr=self.lr_time)

            loss, val_error = train_time_model(time_model, train_dataloader, val_dataloader, time_optim, criterion, self.epoch_time)
            if plot_loss:
                self._plot_loss_error(self.epoch_time, loss, val_error)
            
            self.predictor['time'] = time_model
            self.trained['time'] = True

        elif target == 'energy':
            if (self.trained['time'] == False) and (self.train_power_model_using_true_time == False):
                print("Time model is not trained. First training the time model.")
                self.train('time', plot_loss)

                print("Training power model using the trained time model.")

            power_model = NeuSightMLP(n_layer=self.MLP_layers_power, hidden_dim=self.MLP_hidden_power, input_dim=4, output_dim=self.MLP_odim_power)    
            power_optim = torch.optim.AdamW(power_model.parameters(), lr=self.lr_power)

            loss, val_error = train_power_model(self.predictor['time'], power_model, train_dataloader, val_dataloader, \
                                                power_optim, criterion, self.epoch_power, self.gpu_config['power_cap'], \
                                                direct_power=self.train_power_model_using_true_time)
            if plot_loss:
                self._plot_loss_error(self.epoch_power, loss, val_error)

            self.predictor['energy'] = power_model
            self.trained['energy'] = True

        else:
            raise NotImplementedError("Does not support prediction other than time/energy!")
        
    def _plot_loss_error(self, epochs, loss, error):
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].plot(np.arange(epochs), loss)
        axs[0].set_title('Loss on Train Set')

        error = [x * 100. for x in error]
        axs[1].plot(np.arange(epochs), error)
        axs[1].set_title('Validation Percent Error')
        # axs[1].set_ylim(0, 100)
        plt.show()

    def test(self, target):
        self.test_df['{}_estimate'.format(target)] = -1

        for i, row in self.test_df.iterrows():
            input_feature = np.asarray([row['flop_per_tile'], row['mem_per_tile'], row['mem_dram'], row['mem_l2']])
            if input_feature.ndim < 2:
                input_feature = np.expand_dims(input_feature, axis=0)
            input_feature = input_feature / np.asarray(self.test_norm_vector)

            if self.log_normalize_inputs:
                input_feature = np.log2(input_feature)

            input_feature = torch.Tensor(input_feature)
        
            time_pred = self.predictor['time'](input_feature.float())
            time_alpha = torch.sigmoid(time_pred[:, 0]).item()
            time_beta = torch.sigmoid(time_pred[:, 1]).item()

            wave = row['num_waves']
            roofline = row['roofline_bw']
            flop_per_wave = row['flop_per_wave']
            time_util = time_alpha - time_beta / wave
            ebw = time_util * roofline

            estimated = flop_per_wave / ebw * wave * 1000.
            self.test_df.at[i, 'time_estimate'] = estimated
            time = estimated

            if target == 'time':
                continue
            elif target == 'energy':
                power_util = torch.sigmoid(self.predictor['energy'](input_feature.float())).item()
                ep = power_util * self.gpu_config['power_cap']
                estimated = time * ep / 1000. # ms --> s
                self.test_df.at[i, 'energy_estimate'] = estimated
            else:
                raise TypeError("Unsupported prediction type.")
            
    


            
            





