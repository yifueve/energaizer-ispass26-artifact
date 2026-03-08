import os
import sys

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split

sys.path.append('..')

from tqdm import tqdm

def prec_to_precision_bits(x):
    if (x == 'fp32'):
        return 32
    elif (x == 'fp16') or (x == 'bf16'):
        return 16
    elif (x == 'fp64'):
        return 64
    elif (x == 'int8'):
        return 8
    else:
        raise TypeError("ERROR: Precision {} for GEMM is currently not supported.".format(x))

class NeuSightMLP(nn.Module):
    def __init__(self, n_layer=3, hidden_dim=512, input_dim=4, output_dim=2):
        super().__init__()

        assert(n_layer > 2)

        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.ReLU(inplace=True))

        for i in range(n_layer - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU(inplace=True))

        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x
    
class NeuSightDataset(Dataset):

    def __init__(self, df, normalization_vector, log_norm=False):
        super().__init__()
        self.df = df
        self.normalization_vector = normalization_vector

        self.preprocess(log_norm)
    
    def preprocess(self, log_norm):
        # Construct np.array for input feature and output energy value
        self.input_feature = self.df[['flop_per_tile', 'mem_per_tile', 'mem_dram', 'mem_l2']].to_numpy()
        self.input_feature = self.input_feature / np.asarray(self.normalization_vector)

        if log_norm:
            self.input_feature = np.log2(self.input_feature)

        self.wave = self.df[['num_waves']].to_numpy()
        self.roofline_bw = self.df[['roofline_bw']].to_numpy()

        self.output_time = self.df[['time']].to_numpy()
        self.output_energy = self.df[['energy']].to_numpy()
        self.output_power = self.df[['power']].to_numpy()

        self.flop_per_wave =self.df[['flop_per_wave']].to_numpy()

        self.df['gpu_tdp'] = -1
        self.gpu_tdp = self.df[['gpu_tdp']].to_numpy()

    def __len__(self):
        return self.output_time.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = {'input': self.input_feature[idx], 'output_time': self.output_time[idx], 'output_energy': self.output_energy[idx], \
                  'wave': self.wave[idx], 'roofline_bw': self.roofline_bw[idx], 'flop_per_wave': self.flop_per_wave[idx], \
                  'output_power': self.output_power[idx], 'gpu_tdp': self.gpu_tdp[idx]}

        return sample
    
def MAPELoss(pred, target):
    return torch.mean(torch.abs((target - pred) / target))
def SMAPELoss(pred, target):
    return torch.mean(torch.abs((target - pred) / (target + pred)))

def train_time_model(time_model, train_dataloader, val_dataloader, optim, criterion, epochs):
    time_loss = []
    time_val_error = []

    for epoch in tqdm(range(epochs)):
        time_model = time_model.train()
        curr_time_loss = []
        for batch in train_dataloader:
            batch_input = batch['input']
            batch_output_time = batch['output_time']
            batch_output_energy = batch['output_energy']
            batch_wave = batch['wave']
            batch_roofline = batch['roofline_bw']
            flop_per_wave = batch['flop_per_wave']

            batch_input = batch_input.float()
            batch_output_time = batch_output_time.float()
            batch_output_energy = batch_output_energy.float()

            optim.zero_grad()
            
            time_pred = time_model(batch_input)
            # actual time --> roofline_bw * (alpha - beta / num_wave) = ebw, flops_per_wave / ebw
            time_alpha = torch.sigmoid(time_pred[:, 0]).unsqueeze(-1)
            time_beta = torch.sigmoid(time_pred[:, 1]).unsqueeze(-1)
            time_util = time_alpha - time_beta / (batch_wave)
            time_util = torch.clamp(time_util, min=1e-3)
            ebw = time_util * batch_roofline

            time = flop_per_wave / ebw * batch_wave * 1000. # flops_per_wave is in gflop, ebw is gflops, batch wave is unitless --> second --> *1000: ms
            _time_loss = criterion(time, batch_output_time.reshape(time.shape))
            _time_loss.backward()
            curr_time_loss.append(_time_loss)
            optim.step()
        
        time_loss_sum = sum(curr_time_loss) / len(curr_time_loss)
        time_loss.append(time_loss_sum.detach())

        # validation 
        if val_dataloader is not None:
            time_model = time_model.eval()
            time_pct_error = []
            for batch in val_dataloader:
                batch_input = batch['input']
                batch_output_time = batch['output_time']
                batch_output_energy = batch['output_energy']
                batch_wave = batch['wave']
                batch_roofline = batch['roofline_bw']
                flop_per_wave = batch['flop_per_wave']

                batch_input = batch_input.float()
                batch_output_time = batch_output_time.float()
                batch_output_energy = batch_output_energy.float()
                
                time_pred = time_model(batch_input)
                # actual time --> roofline_bw * (alpha - beta / num_wave) = ebw, flops_per_wave / ebw
                time_alpha = torch.sigmoid(time_pred[:, 0]).unsqueeze(-1)
                time_beta = torch.sigmoid(time_pred[:, 1]).unsqueeze(-1)
                time_util = time_alpha - time_beta / batch_wave
                time_util = torch.clamp(time_util, min=1e-3)
                ebw = time_util * batch_roofline

                time = flop_per_wave / ebw * batch_wave * 1000.
                _time_pct_error = torch.mean(torch.abs((time - batch_output_time) / batch_output_time))
                time_pct_error.append(_time_pct_error)
            
            time_pct_error_sum = sum(time_pct_error) / len(time_pct_error)
            time_val_error.append(time_pct_error_sum.detach())

    return time_loss, time_val_error

def train_power_model(time_model, power_model, train_dataloader, val_dataloader, optim, criterion, epochs, gpu_tdp, direct_power=False):
    power_loss = []
    energy_val_error = []
    if not direct_power:
        time_model = time_model.eval()

    for epoch in tqdm(range(epochs)):
        power_model = power_model.train()
        curr_power_loss = []

        for batch in train_dataloader:
            batch_input = batch['input']
            batch_output_time = batch['output_time']
            batch_output_energy = batch['output_energy']
            batch_wave = batch['wave']
            batch_roofline = batch['roofline_bw']
            flop_per_wave = batch['flop_per_wave']
            batch_output_power = batch['output_power']
            batch_gpu_tdp = batch['gpu_tdp']

            batch_input = batch_input.float()
            batch_output_time = batch_output_time.float()
            batch_output_energy = batch_output_energy.float()
            batch_output_power = batch_output_power.float()
            batch_gpu_tdp = batch_gpu_tdp.float()

            if not direct_power:
            
                time_pred = time_model(batch_input)
                # actual time --> roofline_bw * (alpha - beta / num_wave) = ebw, flops_per_wave / ebw
                time_alpha = torch.sigmoid(time_pred[:, 0]).unsqueeze(-1)
                time_beta = torch.sigmoid(time_pred[:, 1]).unsqueeze(-1)
                time_util = time_alpha - time_beta / (batch_wave)
                time_util = torch.clamp(time_util, min=1e-3)
                ebw = time_util * batch_roofline
                time = flop_per_wave / ebw * batch_wave * 1000. # flops_per_wave is in gflop, ebw is gflops, batch wave is unitless --> second --> *1000: ms
                
                optim.zero_grad()

                power_util = torch.sigmoid(power_model(batch_input))
                # power_alpha = torch.sigmoid(power_pred[:, 0]).unsqueeze(-1)
                # power_util = power_alpha # - power_beta / flop_per_wave
                ep = power_util * gpu_tdp if gpu_tdp > 0 else power_util * batch_gpu_tdp

                energy = ep * time.detach() / 1000. # ms --> s
                _energy_loss = criterion(energy, batch_output_energy.reshape(energy.shape))
                _energy_loss.backward()
                curr_power_loss.append(_energy_loss)
                optim.step()

            else:
                optim.zero_grad()

                power_util = torch.sigmoid(power_model(batch_input))
                ep = power_util * gpu_tdp if gpu_tdp > 0 else power_util * batch_gpu_tdp

                _energy_loss = criterion(ep, batch_output_power.reshape(ep.shape))
                _energy_loss.backward()
                curr_power_loss.append(_energy_loss)
                optim.step()

        power_loss_sum = sum(curr_power_loss) / len(curr_power_loss)
        power_loss.append(power_loss_sum.detach())

        # validation 
        if val_dataloader is not None:
            power_model = power_model.eval()
            energy_pct_error = []

            for batch in val_dataloader:
                batch_input = batch['input']
                batch_output_time = batch['output_time']
                batch_output_energy = batch['output_energy']
                batch_wave = batch['wave']
                batch_roofline = batch['roofline_bw']
                flop_per_wave = batch['flop_per_wave']
                batch_gpu_tdp = batch['gpu_tdp']

                batch_input = batch_input.float()
                batch_output_time = batch_output_time.float()
                batch_output_energy = batch_output_energy.float()
                batch_gpu_tdp = batch_gpu_tdp.float()
                
                if not direct_power:
                    time_pred = time_model(batch_input)
                    # actual time --> roofline_bw * (alpha - beta / num_wave) = ebw, flops_per_wave / ebw
                    time_alpha = torch.sigmoid(time_pred[:, 0]).unsqueeze(-1)
                    time_beta = torch.sigmoid(time_pred[:, 1]).unsqueeze(-1)
                    time_util = time_alpha - time_beta / batch_wave
                    time_util = torch.clamp(time_util, min=1e-3)
                    ebw = time_util * batch_roofline

                    time = flop_per_wave / ebw * batch_wave * 1000.
                else:
                    time = batch_output_time
                
                power_util = torch.sigmoid(power_model(batch_input))
                # power_alpha = torch.sigmoid(power_pred[:, 0]).unsqueeze(-1)
                # power_util = power_alpha # - power_beta / flop_per_wave
                ep = power_util * gpu_tdp if gpu_tdp > 0 else power_util * batch_gpu_tdp

                # print(gpu_tdp, batch_gpu_tdp)

                energy = ep * time.detach() / 1000. # ms --> s
                _energy_pct_error = torch.mean(torch.abs((energy - batch_output_energy) / batch_output_energy))
                energy_pct_error.append(_energy_pct_error)
            
            energy_pct_error_sum = sum(energy_pct_error) / len(energy_pct_error)
            energy_val_error.append(energy_pct_error_sum.detach())

    return power_loss, energy_val_error