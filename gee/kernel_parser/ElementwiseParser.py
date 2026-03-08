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

class ElementwiseParser(BaseKernelParser):
    def __init__(self):
        op = 'elementwise'
        ops_supported = ['pointwise_mul', 'pointwise_add', \
                         'scalar_mul', 'scalar_add', \
                         'typecast_to_fp32', 'typecast_to_bf16', \
                         'relu', 'gelu', 'silu', 'tanh', 'sigmoid', \
                         'unspecified_activation', 'unspecified_tensor', 'unspecified_scalar']
        
        super().__init__(op, ops_supported)

    def parse(self, query, **kwargs):
        parsed = self._parse(query)
        return parsed

    def parse_dataframe(self, df, **kwargs):
        parsed_key_list = [
            'kernel_type', \
            'block_tile', \
            'num_block_tile', \
            'warp_tile', \
            'num_warp_tile', \
            'n_warps_per_block', \
            'total_block_tiles', \
            'elements_per_thread'
        ]
        df[parsed_key_list] = -1
        for idx, row in df.iterrows():
            parsed = self.parse(row)
            for key, value in parsed.items():
                df.loc[idx, key] = value

    def _get_io_size(self, op, prec):

        def prec_to_byte(prec):
            if prec == 'fp32':
                return 4
            elif (prec == 'bf16') or (prec == 'fp16'):
                return 2
            elif (prec == 'int8') or (prec == 'fp8'):
                return 1
            elif (prec == 'fp64'):
                return 8
            else:
                raise NotImplementedError()

        if op in ['pointwise_mul', 'pointwise_add', 'unspecified_tensor']:
            return 3 * prec_to_byte(prec), prec_to_byte(prec)
        elif op == 'typecast_to_fp32':
            return prec_to_byte(prec) + 4, 4
        elif op == 'typecast_to_bf16':
            return prec_to_byte(prec) + 2, 2
        else:
            return 2 * prec_to_byte(prec), prec_to_byte(prec)

    def _parse(self, query):
        # query should have following fields for parsing:
        # 'op', 'dim', 'prec', 'kernel_name', 'grid_size', 'block_size', 'max_concurrent_block'
        # parser returns following field
        # 'kernel_type', 'block_tile', 'num_block_tile', 'warp_tile', 'num_warp_tile', 'n_warps_per_block', 'elements_per_thread'  

        kernel_name = query['kernel_name'].lower()
        block_size = query['block_size']
        if type(block_size) == str:
            block_size = eval(block_size)
        grid_size = query['grid_size']
        if type(grid_size) == str:
            grid_size = eval(grid_size)

        # three meta-kernel types for elementwise operations in pytorch
        #  https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/CUDALoops.cuh#L661
        # - vectorized_elementwise -> threads = 128, elements_per_thread = 8 (16 if io_sizes = 1 -> sum of i+o datatypes is just 1 byte)
        # - unrolled_elementwise -> threads = 128, elements_per_thread = 4
        # - elementwise -> threads = 128, elements_per_thread = 2 if result type has >= 4 bytes else 4

        io_size, o_size = self._get_io_size(query['op'], query['prec'])

        if 'vectorized_elementwise' in kernel_name:
            num_threads_per_block = block_size[0] * block_size[1] * block_size[2]
            assert (num_threads_per_block == 128)

            if io_size == 1:
                elements_per_thread = 16
            else:
                elements_per_thread = 8

            kernel_type = 'vectorized_elementwise'
            
        elif 'unrolled_elementwise' in kernel_name:
            num_threads_per_block = block_size[0] * block_size[1] * block_size[2]
            assert (num_threads_per_block == 128)

            # Exception: A10 (Lambda) -> elements_per_thread = 8 (check torch ver.)
            elements_per_thread = 4
            if grid_size[0] * grid_size[1] * grid_size[2] != (math.ceil(query['dim'] / num_threads_per_block / elements_per_thread)):
                elements_per_thread = 8
            kernel_type = 'unrolled_elementwise'

        elif 'elementwise' in kernel_name:
            num_threads_per_block = block_size[0] * block_size[1] * block_size[2]
            assert (num_threads_per_block == 128)

            if o_size >= 4:
                elements_per_thread = 2
            else:
                elements_per_thread = 4

            kernel_type = 'elementwise'

        else:
            raise NotImplementedError("Unrecognized kernel type! Check kernel name: ", kernel_name)
        
        block_tile = num_threads_per_block * elements_per_thread
        num_block_tile = math.ceil(query['dim'] / block_tile)
        warp_tile = 32 * elements_per_thread
        num_warp_tile = 4
        n_warps_per_block = 4
        total_block_tiles = grid_size[0] * grid_size[1] * grid_size[2]

        if total_block_tiles != num_block_tile:
            print(query['kernel_name'], query['dim'], query['grid_size'])
        assert(total_block_tiles == num_block_tile)

        parsed = {
            'kernel_type': kernel_type, \
            'block_tile': block_tile, \
            'num_block_tile': num_block_tile, \
            'warp_tile': warp_tile, \
            'num_warp_tile': num_warp_tile, \
            'n_warps_per_block': n_warps_per_block, \
            'total_block_tiles': total_block_tiles, \
            'elements_per_thread': elements_per_thread
        }

        return parsed

    def calculate_kernel_info_froom_prediction(self, query):
        # assume that the kernel_type has been provided
        kernel_type = query['kernel_type']

        io_size, o_size = self._get_io_size(query['op'], query['prec'])
        if kernel_type == 'vectorized_elementwise':
            num_threads_per_block = 128
            if io_size == 1:
                elements_per_thread = 16
            else:
                elements_per_thread = 8
        elif kernel_type == 'unrolled_elementwise':
            num_threads_per_block = 128
            elements_per_thread = 4
        elif kernel_type == 'elementwise':
            num_threads_per_block = 128
            if o_size >= 4:
                elements_per_thread = 2
            else:
                elements_per_thread = 4
        else:
            raise NotImplementedError()
    
        block_tile = num_threads_per_block * elements_per_thread
        num_block_tile = math.ceil(query['dim'] / block_tile)
        warp_tile = 32 * elements_per_thread
        num_warp_tile = 4
        n_warps_per_block = 4

        query['block_tile'] = block_tile
        query['num_block_tile'] = num_block_tile
        query['warp_tile'] = warp_tile
        query['num_warp_tile'] = num_warp_tile
        query['n_warps_per_block'] = n_warps_per_block
        query['total_block_tiles'] = num_block_tile
        query['elements_per_thread'] = elements_per_thread

        return query    


