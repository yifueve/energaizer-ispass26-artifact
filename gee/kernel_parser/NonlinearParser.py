import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('..')

import os
import shutil

import re
import math

import numpy as  np
import pandas as pd
import yaml

from gee.kernel_parser.BaseKernelParser import BaseKernelParser

class NonlinearParser(BaseKernelParser):
    def __init__(self):
        op = 'nonlinear'
        ops_supported = ['softmax', 'layernorm']
        super().__init__(op, ops_supported)

    def parse(self, query, op, **kwargs):
        if op == 'softmax':
            parsed = self._parse_softmax(query)
        elif op == 'layernorm':
            parsed = self._parse_layernorm(query)
        else:
            raise NotImplementedError()

        return parsed
    
    def parse_dataframe(self, df, op, **kwargs):
        
        if op == 'softmax':
            parsed_key_list = [
                'block_tile_batch', \
                'block_tile_softmax', \
                'num_block_tile_batch', \
                'num_block_tile_softmax', \
                'warp_tile_batch', \
                'warp_tile_softmax', \
                'num_warp_tile_batch', \
                'num_warp_tile_softmax_spatial', \
                'num_warp_tile_softmax_temporal', \
                'n_warps_per_block', \
                'ilp', \
                'total_block_tiles'
            ]
            df[parsed_key_list] = -1
            for idx, row in df.iterrows():
                parsed = self.parse(row.to_dict(), op)
                for key, value in parsed.items():
                    df.loc[idx, key] = value

            df['kernel_type'] = df['kernel_name'].apply(lambda x: 'softmax_warp_forward' if 'softmax_warp_forward' in x \
                                                                   else ('cunn_SoftMaxForwardSmem' if 'cunn_SoftMaxForwardSmem' in x else 'cunn_SoftMaxForward'))
        
        elif op == 'layernorm':
            parsed_key_list = [
                'block_tile_batch', \
                'block_tile_layernorm', \
                'num_block_tile_batch', \
                'num_block_tile_layernorm', \
                'warp_tile_batch', \
                'warp_tile_layernorm', \
                'num_warp_tile_batch', \
                'num_warp_tile_layernorm_spatial', \
                'num_warp_tile_layernorm_temporal', \
                'n_warps_per_block', \
                'ilp', \
                'total_block_tiles'
            ]
            df[parsed_key_list] = -1
            for idx, row in df.iterrows():
                parsed = self.parse(row.to_dict(), op)
                for key, value in parsed.items():
                    df.loc[idx, key] = value

        else:
            raise NotImplementedError()
    
    
    def _parse_softmax(self, query):

        kernel_name = query['kernel_name'].lower()
        block_size = query['block_size']
        if type(block_size) == str:
            block_size = eval(block_size)
        grid_size = query['grid_size']
        if type(grid_size) == str:
            grid_size = eval(grid_size)
        
        if 'softmax_warp_forward' in kernel_name:
            num_threads_per_block = block_size[0] * block_size[1]

            # batch level information
            num_warp_tile_batch = block_size[1]
            num_block_tile_batch = grid_size[0]
            warp_tile_batch = math.ceil(query['batch'] / (num_warp_tile_batch * num_block_tile_batch))
            block_tile_batch = warp_tile_batch * num_warp_tile_batch

            # softmax level information
            block_tile_softmax = query['dim']
            num_block_tile_softmax = 1
            num_warp_tile_softmax_spatial = 1 # No spatial tiling for softmax dim
            num_warp_tile_softmax_temporal = math.ceil(query['dim'] / 32)
            warp_tile_softmax = query['dim']

            ilp = num_warp_tile_softmax_temporal * warp_tile_batch

            effective_threads = num_threads_per_block
            effective_warps = effective_threads // 32

        # Large softmax dimension, using SMEM or not
        elif ('cunn_softmaxforwardsmem' in kernel_name) or ('cunn_softmaxforward' in kernel_name):
            num_threads_per_block = block_size[0]
            ilp = 8 if query['prec'] in ['bf16', 'fp16'] else 4

            # threadblock-level (all spatial)
            block_tile_batch = 1
            block_tile_softmax = query['dim']
            num_block_tile_batch = query['batch']
            num_block_tile_softmax = 1

            # warp-level
            warp_tile_batch = 1
            num_warp_tile_batch = 1
            num_warp_tile_softmax_temporal = math.ceil(query['dim'] / (ilp * num_threads_per_block))

            effective_threads = min(num_threads_per_block, math.ceil(query['dim'] / ilp))
            effective_warps = effective_threads // 32
            num_warp_tile_softmax_spatial = effective_warps
            warp_tile_softmax = 32 * ilp * num_warp_tile_softmax_temporal
            
        else:
            raise NotImplementedError()
        
        parsed = {
            'block_tile_batch': block_tile_batch, \
            'block_tile_softmax': block_tile_softmax, \
            'num_block_tile_batch': num_block_tile_batch, \
            'num_block_tile_softmax': num_block_tile_softmax, \
            'warp_tile_batch': warp_tile_batch, \
            'warp_tile_softmax': warp_tile_softmax, \
            'num_warp_tile_batch': num_warp_tile_batch, \
            'num_warp_tile_softmax_spatial': num_warp_tile_softmax_spatial, \
            'num_warp_tile_softmax_temporal': num_warp_tile_softmax_temporal, \
            'n_warps_per_block': effective_warps, \
            'ilp': ilp, \
            'total_block_tiles': num_block_tile_batch * num_block_tile_softmax
        }

        return parsed

    def _parse_layernorm(self, query):
        # vectorized_layer_norm_kernel in PyTorch
        # temporal/spatial tiling is similar to cunn_softmaxforward

        kernel_name = query['kernel_name'].lower()
        block_size = query['block_size']
        if type(block_size) == str:
            block_size = eval(block_size)
        grid_size = query['grid_size']
        if type(grid_size) == str:
            grid_size = eval(grid_size)

        num_threads_per_block = block_size[0] * block_size[1] * block_size[2]
        ilp = 4 # vector size is fixed to 4 https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/layer_norm_kernel.cu#L37
        
        block_tile_batch = 1
        block_tile_layernorm = query['dim']
        num_block_tile_batch = query['batch']
        num_block_tile_layernrom = 1

        warp_tile_batch = 1
        num_warp_tile_batch = 1

        num_warp_tile_layernorm_temporal = math.ceil(query['dim'] / (ilp * num_threads_per_block))
        effective_threads = min(num_threads_per_block, math.ceil(query['dim'] / ilp))
        effective_warps = effective_threads // 32
        num_warp_tile_layernorm_spatial = effective_warps
        warp_tile_layernorm = 32 * ilp * num_warp_tile_layernorm_temporal

        parsed = {
            'block_tile_batch': block_tile_batch, \
            'block_tile_layernorm': block_tile_layernorm, \
            'num_block_tile_batch': num_block_tile_batch, \
            'num_block_tile_layernorm': num_block_tile_layernrom, \
            'warp_tile_batch': warp_tile_batch, \
            'warp_tile_layernorm': warp_tile_layernorm, \
            'num_warp_tile_batch': num_warp_tile_batch, \
            'num_warp_tile_layernorm_spatial': num_warp_tile_layernorm_spatial, \
            'num_warp_tile_layernorm_temporal': num_warp_tile_layernorm_temporal, \
            'n_warps_per_block': effective_warps, \
            'ilp': ilp, \
            'total_block_tiles': num_block_tile_batch * num_block_tile_layernrom
        }

        return parsed
    
    def calculate_kernel_info_from_prediction(self, query, op_nonlinear):
        if op_nonlinear == 'softmax':
            # kernel_type, n_warps_per_block, warp_tile_batch provided in kernel_info
            if query['kernel_type'] == 'softmax_warp_forward':
                
                # query['warp_tile_batch'] = math.ceil(query['batch'] / (query['num_warp_tile_batch'] * query['num_block_tile_batch']))
                query['num_warp_tile_batch'] = query['n_warps_per_block']
                query['num_block_tile_batch'] = math.ceil(query['batch'] / (query['warp_tile_batch'] * query['num_warp_tile_batch']))
                query['block_tile_batch'] = query['warp_tile_batch'] * query['num_warp_tile_batch']

                query['block_tile_softmax'] = query['dim']
                query['num_block_tile_softmax'] = 1
                query['num_warp_tile_softmax_spatial'] = 1
                query['num_warp_tile_softmax_temporal'] = math.ceil(query['dim'] / 32)
                query['warp_tile_softmax'] = query['dim']

                query['ilp'] = query['num_warp_tile_softmax_temporal'] * query['warp_tile_batch']
                query['total_block_tiles'] = query['num_block_tile_batch'] * query['num_block_tile_softmax']
            else:
                query['ilp'] = 8 if query['prec'] in ['bf16', 'fp16'] else 4

                query['block_tile_batch'] = 1
                query['block_tile_softmax'] = query['dim']
                query['num_block_tile_batch'] = query['batch']
                query['num_block_tile_softmax'] = 1

                query['num_warp_tile_batch'] = 1
                query['num_warp_tile_softmax_temporal'] = math.ceil(query['dim'] / (query['ilp'] * query['n_warps_per_block'] * 32))
                query['num_warp_tile_softmax_spatial'] = min((query['n_warps_per_block'] * 32), math.ceil(query['dim'] / query['ilp'])) // 32
                query['warp_tile_softmax'] = 32 * query['ilp'] * query['num_warp_tile_softmax_temporal']
                query['total_block_tiles'] = query['num_block_tile_batch'] * query['num_block_tile_softmax']

        elif op_nonlinear == 'layernorm':
            query['ilp'] = 4
            query['n_warps_per_block'] = 4 # fixed to 128 threads / block (https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/thread_constants.h#L19)

            query['block_tile_batch'] = 1
            query['block_tile_layernorm'] = query['dim']
            query['num_block_tile_batch'] = query['batch']
            query['num_block_tile_layernorm'] = 1

            query['num_warp_tile_batch'] = 1
            query['num_warp_tile_layernorm_temporal'] = math.ceil(query['dim'] / (query['ilp'] * query['n_warps_per_block'] * 32))
            query['num_warp_tile_layernorm_spatial'] = min((query['n_warps_per_block'] * 32), math.ceil(query['dim'] / query['ilp']))
            query['warp_tile_layernorm'] = 32 * query['ilp'] * query['num_warp_tile_layernorm_temporal']
            query['total_block_tiles'] = query['num_block_tile_batch'] * query['num_block_tile_layernorm']

            query['warp_tile_batch'] = 1
            query['n_warps_per_block'] = 4

        else:
            raise NotImplementedError()
        
        return query
