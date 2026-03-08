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
import copy

from gee.kernel_parser.BaseKernelParser import BaseKernelParser

tensorop_warp_shape_dict = {}

tensorop_warp_shape_dict['884'] = [(32, 32, 4), (32, 64, 4), (64, 32, 4), (64, 64, 4)]
tensorop_warp_shape_dict['1688'] = [(32, 32, 8), (32, 64, 8), (64, 32, 8), (64, 64, 8)]
tensorop_warp_shape_dict['16816'] = [(32, 32, 16), (32, 64, 16), (64, 32, 16), (64, 64, 16)]
tensorop_warp_shape_dict['8816'] = [(32, 32, 16), (32, 64, 16), (64, 32, 16), (64, 64, 16)]
tensorop_warp_shape_dict['8832'] = [(32, 32, 32), (32, 64, 32), (64, 32, 32), (64, 64, 32)]
tensorop_warp_shape_dict['16832'] = [(32, 32, 32), (32, 64, 32), (64, 32, 32), (64, 64, 32)]
tensorop_warp_shape_dict['16864'] = [(32, 32, 64), (32, 64, 64), (64, 32, 64), (64, 64, 64)]
tensorop_warp_shape_dict['88128'] = [(32, 32, 128), (32, 64, 128), (64, 32, 128), (64, 64, 128)]
tensorop_warp_shape_dict['168256'] = [(32, 32, 256), (32, 64, 256), (64, 32, 256), (64, 64, 256)]
tensorop_warp_shape_dict['161616'] = [(32, 32, 16), (32, 64, 16), (64, 32, 16)]
tensorop_warp_shape_dict['83216'] = [(32, 32, 16), (32, 64, 16), (64, 32, 16)]
tensorop_warp_shape_dict['32816'] = [(32, 32, 16), (32, 64, 16), (64, 32, 16)]

tensorop_math_inst_dict = {}

tensorop_math_inst_dict['884'] = (8, 8, 4)
tensorop_math_inst_dict['1688'] = (16, 8, 8)
tensorop_math_inst_dict['16816'] = (16, 8, 16)
tensorop_math_inst_dict['8816'] = (8, 8, 16)
tensorop_math_inst_dict['8832'] = (8, 8, 32)
tensorop_math_inst_dict['16832'] = (16, 8, 32)
tensorop_math_inst_dict['16864'] = (16, 8, 64)
tensorop_math_inst_dict['88128'] = (8, 8, 128)
tensorop_math_inst_dict['168256'] = (16, 8, 256)
tensorop_math_inst_dict['161616'] = (16, 16, 16)
tensorop_math_inst_dict['83216'] = (8, 32, 16)
tensorop_math_inst_dict['32816'] = (32, 8, 16)

kernels_using_cuda = {'gemv2t', 'gemvnsp'}

class GemmLikeParser(BaseKernelParser):
    def __init__(self):
        op = 'gemm_like'
        ops_supported = ['gemm', 'fmha-approximate']

        super().__init__(op, ops_supported)
        
    def parse(self, query, op, **kwargs):
        if op == 'gemm':
            # information to be returned
            parsed = {
                'use_cuda_core_only': False, \
                'gemv': False, \
                'block_tile_M': -1, \
                'block_tile_N': -1, \
                'block_tile_K': -1, \
                'num_block_tile_batch': -1, \
                'num_block_tile_M': -1, \
                'num_block_tile_N': -1, \
                'num_block_tile_K': -1, \
                'total_block_tiles': -1, \
                'splitK': -1, \
                'totalK': -1, \
                'splitK_batch': -1, \
                'stagesK': -1, \
                'multistageK': -1, \
                'threads': -1, \
                'n_warps_per_block': -1, \
                'warp_tile_M': -1, \
                'warp_tile_N': -1, \
                'warp_tile_K': -1, \
                'num_warp_tile_M': -1, \
                'num_warp_tile_N': -1, \
                'num_warp_tile_K': -1, \
                'math_inst_M': -1, \
                'math_inst_N': -1, \
                'math_inst_K': -1, \
                'sliceK': -1, \
                'groupsK': -1
            }
            self._parse_cublas_gemm_block(query, parsed)

            for k, v in parsed.items():
                query[k] = v

            self._paser_cublas_gemm_warp(query, parsed)

        else:
            raise NotImplementedError()

        return parsed
    
    def parse_dataframe(self, df, op, **kwargs):
        if op == 'gemm':
            parsed_key_list = [
            'use_cuda_core_only', \
            'gemv', \
            'block_tile_M', \
            'block_tile_N', \
            'block_tile_K', \
            'num_block_tile_batch', \
            'num_block_tile_M', \
            'num_block_tile_N', \
            'num_block_tile_K', \
            'total_block_tiles', \
            'splitK', \
            'totalK', \
            'splitK_batch', \
            'stagesK', \
            'multistageK', \
            'threads', \
            'n_warps_per_block', \
            'warp_tile_M', \
            'warp_tile_N', \
            'warp_tile_K', \
            'num_warp_tile_M', \
            'num_warp_tile_N', \
            'num_warp_tile_K', \
            'math_inst_M', \
            'math_inst_N', \
            'math_inst_K', \
            'sliceK', \
            'groupsK'
            ]
            df[parsed_key_list] = -1
            for idx, row in df.iterrows():
                # information to be returned
                parsed = {
                    'use_cuda_core_only': False, \
                    'gemv': False, \
                    'block_tile_M': -1, \
                    'block_tile_N': -1, \
                    'block_tile_K': -1, \
                    'num_block_tile_batch': -1, \
                    'num_block_tile_M': -1, \
                    'num_block_tile_N': -1, \
                    'num_block_tile_K': -1, \
                    'total_block_tiles': -1, \
                    'splitK': -1, \
                    'totalK': -1, \
                    'splitK_batch': -1, \
                    'stagesK': -1, \
                    'multistageK': -1, \
                }
                self._parse_cublas_gemm_block(row.to_dict(), parsed)
                for key, value in parsed.items():
                    df.loc[idx, key] = value
            
            # Post-processing for gemv
            kernels = df['kernel_name'].unique()
            gemv_kernels = [x for x in kernels if ('gemv' in x.lower()) or ('largek' in x.lower())]

            unique_combinations = df[['kernel_name', 'grid_size']].drop_duplicates().values.tolist()
            for (name, grid) in unique_combinations:
                if name not in gemv_kernels:
                    continue
                _df = df.loc[(df['kernel_name'] == name) & (df['grid_size'] == grid)]
                block_tile_M = math.ceil(np.max(_df['dimM'] / _df['num_block_tile_M']))
                block_tile_N = math.ceil(np.max(_df['dimN'] / _df['num_block_tile_N']))
                df.loc[(df['kernel_name'] == name) & (df['grid_size'] == grid), 'block_tile_M'] = block_tile_M
                df.loc[(df['kernel_name'] == name) & (df['grid_size'] == grid), 'block_tile_N'] = block_tile_N

            # Warp 
            for idx, row in df.iterrows():
                parsed = {
                    'threads': -1, \
                    'n_warps_per_block': -1, \
                    'warp_tile_M': -1, \
                    'warp_tile_N': -1, \
                    'warp_tile_K': -1, \
                    'num_warp_tile_M': -1, \
                    'num_warp_tile_N': -1, \
                    'num_warp_tile_K': -1, \
                    'math_inst_M': -1, \
                    'math_inst_N': -1, \
                    'math_inst_K': -1, \
                    'sliceK': -1, \
                    'groupsK': -1
                }
                self._paser_cublas_gemm_warp(row.to_dict(), parsed)
                for key, value in parsed.items():
                    df.loc[idx, key] = value


        else:
            raise NotImplementedError()
        
    def _parse_cublas_gemm_block(self, query, parsed):
        # from query, get kernel_name, grid_size, block_size
        kernel_name = query['kernel_name'].lower()
        block_size = query['block_size']
        if type(block_size) == str:
            block_size = eval(block_size)
        grid_size = query['grid_size']
        if type(grid_size) == str:
            grid_size = eval(grid_size)

        # determine gemv
        if 'gemv' in kernel_name:
            parsed['gemv'] = True

            # SplitK possibility?
            #  - grid_size[2] != 1
            #  - after simply dividing, get too small block_tile_size 
            possible_splitk = (grid_size[2] != 1) and (grid_size[2] == query['batch'])
            parsed['total_block_tiles'] = grid_size[0] * grid_size[1] * grid_size[2]

            # Special case: M=1 or N=1 (only one tile for a dimension with size 1)
            if (query['dimM'] == 1) or (query['dimN'] == 1):
                parsed['num_block_tile_batch'] = query['batch']

                if not possible_splitk:
                    if (query['dimM'] == 1):
                        parsed['num_block_tile_M'] = 1
                        parsed['num_block_tile_N'] = int(parsed['total_block_tiles'] / parsed['num_block_tile_batch'])
                    elif (query['dimN'] == 1):
                        parsed['num_block_tile_M'] = int(parsed['total_block_tiles'] / parsed['num_block_tile_batch'])
                        parsed['num_block_tile_N'] = 1
                else:
                    if (query['dimM'] == 1):
                        parsed['num_block_tile_M'] = 1
                        parsed['num_block_tile_N'] = grid_size[0] * grid_size[1]
                    elif (query['dimN'] == 1):
                        parsed['num_block_tile_M'] = grid_size[0] * grid_size[1]
                        parsed['num_block_tile_N'] = 1
            
            # General GEMV -> grid size corresponds to M, N, batch dimension
            else:
                parsed['num_block_tile_M'] = grid_size[0]
                parsed['num_block_tile_N'] = grid_size[1]
                parsed['num_block_tile_batch'] = query['batch']

            parsed['block_tile_M'] = math.ceil(query['dimM'] / parsed['num_block_tile_M'])
            parsed['block_tile_N'] = math.ceil(query['dimN'] / parsed['num_block_tile_N'])

            if parsed['total_block_tiles'] != (parsed['num_block_tile_batch'] * parsed['num_block_tile_M'] * parsed['num_block_tile_N']):
                num_block_tile = parsed['num_block_tile_batch'] * parsed['num_block_tile_M'] * parsed['num_block_tile_N']
                num_block_tile_from_grid = grid_size[0] * grid_size[1] * grid_size[2]
                parsed['splitK_batch'] = int(num_block_tile_from_grid / num_block_tile)
                parsed['splitK'] = True
                parsed['totalK'] = math.ceil(query['dimK'] / parsed['splitK_batch'])
                parsed['num_block_tile_batch'] *= parsed['splitK_batch']
                parsed['splitK'] = True
            else:
                parsed['splitK'] = False
                parsed['totalK'] = query['dimK']
                parsed['splitK_batch'] = -1

            parsed['block_tile_K'] = min(32 if query['useTensorCore'] else 8, query['dimK'])
            parsed['stagesK'] = math.ceil(parsed['totalK'] / parsed['block_tile_K'])
            parsed['multistageK'] = 2
            # parsed['num_block_tile_K'] = math.ceil(parsed['totalK'] / parsed['block_tile_K'])

            for special_kernel in kernels_using_cuda:
                if (special_kernel in kernel_name):
                    parsed['use_cuda_core_only'] = True
                    break
            
            if ((query['dimM'] == 1) or (query['dimN'] == 1)) and (query['dimK'] == 1):
                parsed['use_cuda_core_only'] = True

        # largek
        elif 'largek' in kernel_name:
            parsed['gemv'] = True
            parsed['total_block_tiles'] = grid_size[0] * grid_size[1] * grid_size[2]
            parsed['num_block_tile_batch'] = query['batch'] * grid_size[2]

            if (query['dimM'] < query['dimN']):
                # dimN is the tiled dim
                parsed['num_block_tile_M'] = 1
                parsed['num_block_tile_N'] = grid_size[0] * grid_size[1]
            else:
                parsed['num_block_tile_M'] = grid_size[0] * grid_size[1]
                parsed['num_block_tile_N'] = 1
            
            parsed['block_tile_M'] = math.ceil(query['dimM'] / parsed['num_block_tile_M'])
            parsed['block_tile_N'] = math.ceil(query['dimN'] / parsed['num_block_tile_N'])
            parsed['splitK'] = True
            parsed['totalK'] = math.ceil(query['dimK'] / grid_size[2]) 
            parsed['splitK_batch'] = grid_size[2]
            parsed['block_tile_K'] = min(32 if query['useTensorCore'] else 8, query['dimK'])
            parsed['stagesK'] = math.ceil(parsed['totalK'] / parsed['block_tile_K'])
            parsed['multistageK'] = 2
            parsed['use_cuda_core_only'] = False
            # parsed['num_block_tile_K'] = math.ceil(parsed['totalK'] / parsed['block_tile_K'])

        # magma_sgemmEx (bf16 forced to use CUDA cores)
        elif 'magma_sgemmex' in kernel_name:
            parsed['gemv'] = False
            parsed['total_block_tiles'] = grid_size[0] * grid_size[1] * grid_size[2]
            parsed['num_block_tile_batch'] = query['batch']

            if (query['dimM'] < query['dimN']):
                parsed['num_block_tile_M'] = 1
                parsed['num_block_tile_N'] = int(parsed['total_block_tiles'] / query['batch'])
            else:
                parsed['num_block_tile_M'] = int(parsed['total_block_tiles'] / query['batch'])
                parsed['num_block_tile_N'] = 1

            parsed['block_tile_M'] = math.ceil(query['dimM'] / parsed['num_block_tile_M'])
            parsed['block_tile_N'] = math.ceil(query['dimN'] / parsed['num_block_tile_N'])
            parsed['splitK'] = False
            parsed['totalK'] = query['dimK']
            parsed['splitK_batch'] = -1
            parsed['block_tile_K'] = min(32 if query['useTensorCore'] else 8, query['dimK'])
            parsed['stagesK'] = math.ceil(parsed['totalK'] / parsed['block_tile_K'])
            parsed['multistageK'] = 2
            parsed['use_cuda_core_only'] = True

        # regular gemms
        else:
            parsed['gemv'] = False

            # Find threadblock-level tiling information
            res = re.findall('[0-9]+x[0-9]+', kernel_name)
            block_tile_shape_str = res[0].split('x')
            parsed['block_tile_M'] = eval(block_tile_shape_str[0])
            parsed['block_tile_N'] = eval(block_tile_shape_str[1])
            parsed['num_block_tile_M'] = math.ceil(query['dimM'] / parsed['block_tile_M'])
            parsed['num_block_tile_N'] = math.ceil(query['dimN'] / parsed['block_tile_N'])
            parsed['num_block_tile_batch'] = query['batch']

            # Determine split-K: if number of blocks we parsed doesn't match the actual grid size, splitK
            num_block_tile = parsed['num_block_tile_batch'] * parsed['num_block_tile_M'] * parsed['num_block_tile_N']
            num_block_tile_from_grid = grid_size[0] * grid_size[1] * grid_size[2]

            if num_block_tile != num_block_tile_from_grid:
                parsed['splitK_batch'] = int(num_block_tile_from_grid / num_block_tile)
                if parsed['splitK_batch'] == 0:
                    # try transposing block_tile_m and block_tile_n
                    parsed['block_tile_M'] = eval(block_tile_shape_str[1])
                    parsed['block_tile_N'] = eval(block_tile_shape_str[0])
                    parsed['num_block_tile_M'] = math.ceil(query['dimM'] / parsed['block_tile_M'])
                    parsed['num_block_tile_N'] = math.ceil(query['dimN'] / parsed['block_tile_N'])
                    parsed['num_block_tile_batch'] = query['batch']

                    # Determine split-K: if number of blocks we parsed doesn't match the actual grid size, splitK
                    num_block_tile = parsed['num_block_tile_batch'] * parsed['num_block_tile_M'] * parsed['num_block_tile_N']
                    num_block_tile_from_grid = grid_size[0] * grid_size[1] * grid_size[2]
                    parsed['splitK_batch'] = int(num_block_tile_from_grid / num_block_tile)
                    
                parsed['totalK'] = math.ceil(query['dimK'] / parsed['splitK_batch'])
                parsed['num_block_tile_batch'] *= parsed['splitK_batch']
                parsed['splitK'] = True
            else:
                parsed['totalK'] = query['dimK']
                parsed['splitK'] = False

            # Get K-dimension information if possible
            # If sliceK, usually kernel name includes 'sliced' followed by how they slice inside one threadblock
            if (('sliced' in kernel_name) and len(res) > 2) or (('sliced' not in kernel_name) and len(res) > 1):
                parsed['block_tile_K'] = eval(res[-1].split('x')[0])
                parsed['multistageK'] = eval(res[-1].split('x')[1])
                parsed['stagesK'] = math.ceil(parsed['totalK'] / parsed['block_tile_K'])

            # If this information doesn't exist set to default value (8 for cuda, 32 for tensor)
            if parsed['block_tile_K'] < 0:
                parsed['block_tile_K'] = min(32 if query['useTensorCore'] else 8, query['dimK'])
                parsed['stagesK'] = math.ceil(parsed['totalK'] / parsed['block_tile_K'])
            
            # Default multi-stage is just double-buffering
            if parsed['multistageK'] < 0:
                parsed['multistageK'] = 2

        parsed['num_block_tile_K'] = math.ceil(parsed['totalK'] / parsed['block_tile_K'])
        parsed['total_block_tiles'] = parsed['num_block_tile_batch'] * parsed['num_block_tile_M'] * parsed['num_block_tile_N']

    def _paser_cublas_gemm_warp(self, query, parsed):
        # from query, get kernel_name, grid_size, block_size
        kernel_name = query['kernel_name'].lower()
        block_size = query['block_size']
        if type(block_size) == str:
            block_size = eval(block_size)
        grid_size = query['grid_size']
        if type(grid_size) == str:
            grid_size = eval(grid_size)

        # get warp/instruction information
        parsed['threads'] = block_size[0] * block_size[1] * block_size[2]

        num_warp_tile = (-1, -1, -1)
        kernel_warp_tile = (-1, -1, -1)

        res = re.findall('[s|h|i][0-9]+gemm', kernel_name)
        parsed['n_warps_per_block'] = int(parsed['threads'] / 32)

        # find the warp tile candidates (not visible in the kernel name in cublas)
        if query['useTensorCore'] and (not query['use_cuda_core_only']):

            # get tensorcore math instruction shape
            if len(res) == 1:
                math_inst_shape = res[0][1:-4]
            else:
                math_inst_shape = '16816'
                if query['gemv'] and (query['block_tile_K'] < 8):
                    math_inst_shape = '1688'

            # get warptile shape candidates
            warp_shape_candidates = copy.deepcopy(tensorop_warp_shape_dict[math_inst_shape])
            math_inst_shape_decoded = copy.deepcopy(tensorop_math_inst_dict[math_inst_shape])

            # exception for gemv: warp shape is simply the math instruction shape
            if query['gemv']:
                if query['block_tile_K'] < 8:
                    warp_shape_candidates.append((16, 8, 8))
                else:
                    warp_shape_candidates.append((16, 8, 16))
        
        # cuda core -> guess as the warp tile shape that will minimize shared memory traffic
        else:
            math_inst_shape_decoded = (1, 1, 1)

            def find_factor_pairs(number):
                factors = []
                for i in range(1, int(number**0.5) + 1):
                    if number % i == 0:
                        factors.append((i, number // i))
                return factors

            warp_num_candidates_all = find_factor_pairs(parsed['n_warps_per_block'])
            warp_shape_candidates_all = [(math.ceil(query['block_tile_M'] / x[0]), math.ceil(query['block_tile_N'] / x[1]), 1) for x in warp_num_candidates_all]
            effective_block_tiles = [(x[0] * query['block_tile_M'], x[1] * query['block_tile_N'], x[2] * query['block_tile_K']) for x in warp_shape_candidates_all]
            estimated_smem_to_reg = [(effective_block_tiles[i][0] * effective_block_tiles[i][2] * x[1] + effective_block_tiles[i][1] * effective_block_tiles[i][2] * x[0]) for i, x in enumerate(warp_num_candidates_all)]
            minidx = estimated_smem_to_reg.index(min(estimated_smem_to_reg))
            warp_shape_candidates = [warp_shape_candidates_all[minidx]]

        # First, assume there is no 'slice-K' (tiling across K-dimension in different warps) and check if there exists a legal one
        num_warps_per_block_candidates = [math.ceil(query['block_tile_M'] / x[0]) * math.ceil(query['block_tile_N'] / x[1]) for x in warp_shape_candidates]
        warp_shape_candidates_filtered = [warp_shape_candidates[i] for i in range(len(warp_shape_candidates)) if (num_warps_per_block_candidates[i] == parsed['n_warps_per_block'])] #  and (num_inst_per_warp_candidates[i] == tensor_op_inst_per_warp)
        
        # If there is no legal candidate, treat this as 'slice-k'
        if len(warp_shape_candidates_filtered) == 0:
            slicek_candidates_index =[i for i in range(len(warp_shape_candidates)) if (warp_shape_candidates[i][0] <= query['block_tile_M']) and (warp_shape_candidates[i][1] <= query['block_tile_N'])]
            if len(slicek_candidates_index) == 0:
                slicek_candidates_index = [i for i in range(len(warp_shape_candidates))] # relax
                
            num_slices_needed = [parsed['n_warps_per_block'] / num_warps_per_block_candidates[i] for i in slicek_candidates_index]

            min_slices_needed = min(num_slices_needed)
            min_slices_idx = slicek_candidates_index[num_slices_needed.index(min_slices_needed)]
            kernel_warp_tile = warp_shape_candidates[min_slices_idx]

        # If there is a legal candidate
        else:
            if len(warp_shape_candidates_filtered) > 1:
                effective_block_tiles = [(x[0] * query['block_tile_M'], x[1] * query['block_tile_N'], x[2] * query['block_tile_K']) for x in warp_shape_candidates_filtered]
                estimated_smem_to_reg = [(effective_block_tiles[i][0] * effective_block_tiles[i][2] * x[1] + effective_block_tiles[i][1] * effective_block_tiles[i][2] * x[0]) for i, x in enumerate(warp_shape_candidates_filtered)]
                minidx = estimated_smem_to_reg.index(min(estimated_smem_to_reg))
            else:
                minidx = 0

            kernel_warp_tile = warp_shape_candidates_filtered[minidx]
            min_slices_needed = 1

        num_warp_tile = (math.ceil(query['block_tile_M'] / kernel_warp_tile[0]), math.ceil(query['block_tile_N'] / kernel_warp_tile[1]), min_slices_needed)
        groupsK = math.ceil(math.ceil(query['block_tile_K'] / num_warp_tile[2]) / kernel_warp_tile[2])

        parsed['warp_tile_M'] = kernel_warp_tile[0]
        parsed['warp_tile_N'] = kernel_warp_tile[1]
        parsed['warp_tile_K'] = kernel_warp_tile[2]
        parsed['num_warp_tile_M'] = num_warp_tile[0]
        parsed['num_warp_tile_N'] = num_warp_tile[1]
        parsed['num_warp_tile_K'] = num_warp_tile[2]
        parsed['math_inst_M'] = math_inst_shape_decoded[0]
        parsed['math_inst_N'] = math_inst_shape_decoded[1]
        parsed['math_inst_K'] = math_inst_shape_decoded[2]
        parsed['sliceK'] = (num_warp_tile[2] > 1)
        parsed['groupsK'] = groupsK
    
    def calculate_kernel_info_from_prediction(self, query, kernel_info, predicted_splitk, lut):
        # fields = ['block_tile_M', 'block_tile_N', 'block_tile_K', \
        #           'warp_tile_M', 'warp_tile_N', 'warp_tile_K', \
        #           'math_inst_M', 'math_inst_N', 'math_inst_K', \
        #           'gemv', 'use_cuda_core_only', 'multistageK', 'threads', 'max_concurrent_block']

        query_num_block_tile_batch = query['batch']
        query_num_block_tile_M = math.ceil(query['dimM'] / query['block_tile_M'])
        query_num_block_tile_N = math.ceil(query['dimN'] / query['block_tile_N'])
        query_totalK = query['dimK']

       
        predicted_splitk_batch = -1
        if predicted_splitk:
            ref = copy.deepcopy(lut)
            cond = pd.Series(True, index=ref.index)
            for k, v in kernel_info.items():
                cond = cond & (ref[k] == v)
            ref = ref.loc[cond]
            ref = ref.loc[ref['splitK'] == True]
            if len(ref) == 0:
                predicted_splitk_batch = -1
            else:
                ref['original_block_tiles'] = ref['batch'] * ref['num_block_tile_M'] * ref['num_block_tile_N']
                ref['block_tiles_diff'] = np.abs(ref['original_block_tiles'] - query['batch'] * query_num_block_tile_M * query_num_block_tile_N)
                ref['dimK_diff'] = np.abs(ref['dimK']-query['dimK'])
                ref.sort_values(by=['block_tiles_diff', 'dimK_diff'], ascending=[True, True], inplace=True)
                predicted_splitk_batch = ref.iloc[0]['splitK_batch']
            del ref
        
        query['splitK'] = predicted_splitk and (predicted_splitk_batch > 0)
        query['splitK_batch'] = predicted_splitk_batch

        if query['splitK_batch'] > 0:
            query_totalK = math.ceil(query['dimK'] / query['splitK_batch'])
            query_num_block_tile_batch *= query['splitK_batch']

        query_stagesK = math.ceil(query_totalK / query['block_tile_K'])
        query_num_block_tile_K = query_stagesK
        query_total_block_tiles = query_num_block_tile_batch * query_num_block_tile_M * query_num_block_tile_N

        query['num_block_tile_batch'] = query_num_block_tile_batch
        query['num_block_tile_M'] = query_num_block_tile_M
        query['num_block_tile_N'] = query_num_block_tile_N
        query['num_block_tile_K'] = query_num_block_tile_K
        query['totalK'] = query_totalK
        query['stagesK'] = query_stagesK
        query['total_block_tiles'] = query_total_block_tiles

        query['n_warps_per_block'] = int(query['threads'] / 32)
        query['num_warp_tile_M'] = math.ceil(query['block_tile_M'] / query['warp_tile_M'])
        query['num_warp_tile_N'] = math.ceil(query['block_tile_N'] / query['warp_tile_N'])
        
        warps_calculated = query['num_warp_tile_M'] * query['num_warp_tile_N']
        if (query['n_warps_per_block'] > warps_calculated):
            query['num_warp_tile_K'] = int(query['n_warps_per_block'] / warps_calculated)
            query['sliceK'] = True
        else:
            query['num_warp_tile_K'] = 1
            query['sliceK'] = False
        
        query['groupsK'] = int(query['block_tile_K'] / (query['num_warp_tile_K'] * query['math_inst_K']))

        return query

    
    

# if __name__ == '__main__':
#     t = GemmLikeParser()
#     query = {}
#     query['kernel_name'] = 'void cutlass::Kernel<cutlass_80_tensorop_bf16_s16816gemm_bf16_64x256_32x4_nn_align8>(T1::Params)'
#     query['batch'] = 128
#     query['dimM'] = 1024
#     query['dimN'] = 1024
#     query['dimK'] = 128
#     query['grid_size'] = (32, 2, 128)
#     query['block_size'] = (128, 1, 1)
#     query['useTensorCore'] = True
#     parsed = t.parse(query, 'gemm')

#     for key, value in parsed.items():
#         print("{}: {}".format(key, value))