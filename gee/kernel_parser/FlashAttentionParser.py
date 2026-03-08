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

class FlashAttentionParser(BaseKernelParser):
    def __init__(self):
        op = 'flashattention'
        ops_supported = ['flashattention_v2']

        super().__init__(op, ops_supported)

    def parse(self, query, op, **kwargs):
        if op == 'flashattention_v2':
            parsed = {
                'block_r': -1,
                'block_c': -1,
                'num_block_r': -1,
                'num_block_c': -1, 
                'num_block_tile_batch': -1,
                'total_block_tiles': -1, 
                'iteration_stages': -1,
                'multistageK': -1, 
                'threads': -1, 
                'n_warps_per_block': -1, 
                'gemm_warp_tile_M': -1,
                'gemm_warp_tile_N': -1, 
                'gemm_warp_tile_K': -1, 
                'num_gemm_warp_tile_M': -1,
                'num_gemm_warp_tile_N': -1, 
                'num_gemm_warp_tile_K': -1, 
                'gemm_math_inst_M': -1, 
                'gemm_math_inst_N': -1,
                'gemm_math_inst_K': -1, 
                'sliceK': -1, 
                'groupsK': -1
            }
            self._parse_flashattn_block(query, parsed)
            for k, v in parsed.items():
                query[k] = v
            self._parse_flashattn_warp(query, parsed)
            for k, v in parsed.items():
                query[k] = v
        else:
            return NotImplementedError()

        return parsed
    
    def parse_dataframe(self, df, op, **kwargs):
        if op == 'flashattention_v2':
            parsed_key_list = [
                'block_r',
                'block_c',
                'num_block_r',
                'num_block_c', 
                'num_block_tile_batch',
                'total_block_tiles', 
                'iteration_stages',
                'multistageK', 
                'threads', 
                'n_warps_per_block', 
                'QK_gemm_warp_tile_M',
                'QK_gemm_warp_tile_N', 
                'QK_gemm_warp_tile_K', 
                'QK_num_gemm_warp_tile_M',
                'QK_num_gemm_warp_tile_N', 
                'QK_num_gemm_warp_tile_K', 
                'SV_gemm_warp_tile_M',
                'SV_gemm_warp_tile_N', 
                'SV_gemm_warp_tile_K', 
                'SV_num_gemm_warp_tile_M',
                'SV_num_gemm_warp_tile_N', 
                'SV_num_gemm_warp_tile_K', 
                'gemm_math_inst_M', 
                'gemm_math_inst_N',
                'gemm_math_inst_K', 
                'QK_sliceK', 
                'QK_groupsK',
                'SV_sliceK', 
                'SV_groupsK'
            ]
            df[parsed_key_list] = -1
            for idx, row in df.iterrows():
                parsed = self.parse(row.to_dict(), op)
                for key, value in parsed.items():
                    df.loc[idx, key] = value
        
        else:
            raise NotImplementedError()
    
    def _parse_flashattn_block(self, query, parsed):

        # Query should have following fields: batch, n_head, seq_len, head_dim

        # from query, get kernel_name, grid_size, block_size
        kernel_name = query['kernel_name'].lower()
        block_size = query['block_size']
        if type(block_size) == str:
            block_size = eval(block_size)
        grid_size = query['grid_size']
        if type(grid_size) == str:
            grid_size = eval(grid_size)

        if 'flash' in kernel_name:
            flash_attn_kernel_param = re.findall('kernel_traits<\(int\)(\d+),\s*\(int\)(\d+),\s*\(int\)(\d+),\s*\(int\)(\d+),', kernel_name)[0]
            kBlockM = int(flash_attn_kernel_param[1])
            kBlockN = int(flash_attn_kernel_param[2])
            kNWarps = int(flash_attn_kernel_param[3])

            parsed['block_r'] = kBlockM
            parsed['block_c'] = kBlockN
            parsed['n_warps_per_block'] = kNWarps

            parsed['num_block_r'] = int(query['seq_len'] / parsed['block_r'])
            parsed['num_block_c'] = int(query['seq_len'] / parsed['block_c'])
            parsed['num_block_tile_batch'] = int(query['batch'] * query['n_head'])
            parsed['total_block_tiles'] = parsed['num_block_tile_batch'] * parsed['num_block_r']
            parsed['iteration_stages'] = parsed['num_block_c']
            parsed['multistageK'] = 2

        else:
            print("This is not a flash attention kernel!")
            print("Query info: ", query)

    def _parse_flashattn_warp(self, query, parsed, know_threads_warps=False):

        if not know_threads_warps:
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

        # assume that tensorcore instruction shape is (16, 8, 16) for fp/bf16
        math_inst_shape = '16816'
        warp_shape_candidates = copy.deepcopy(tensorop_warp_shape_dict[math_inst_shape])
        math_inst_shape_decoded = copy.deepcopy(tensorop_math_inst_dict[math_inst_shape])

        # First, assume there is no 'slice-K' (tiling across K-dimension in different warps) and check if there exists a legal one
        num_warps_per_block_candidates = [math.ceil(query['block_r'] / x[0]) * math.ceil(query['block_c'] / x[1]) for x in warp_shape_candidates]
        warp_shape_candidates_filtered = [warp_shape_candidates[i] for i in range(len(warp_shape_candidates)) if (num_warps_per_block_candidates[i] == parsed['n_warps_per_block'])] #  and (num_inst_per_warp_candidates[i] == tensor_op_inst_per_warp)
        
        # If there is no legal candidate, treat this as 'slice-k'
        if len(warp_shape_candidates_filtered) == 0:
            slicek_candidates_index =[i for i in range(len(warp_shape_candidates)) if (warp_shape_candidates[i][0] <= query['block_r']) and (warp_shape_candidates[i][1] <= query['block_c'])]
            if len(slicek_candidates_index) == 0:
                slicek_candidates_index = [i for i in range(len(warp_shape_candidates))] # relax
                
            num_slices_needed = [parsed['n_warps_per_block'] / num_warps_per_block_candidates[i] for i in slicek_candidates_index]

            min_slices_needed = min(num_slices_needed)
            min_slices_idx = slicek_candidates_index[num_slices_needed.index(min_slices_needed)]
            kernel_warp_tile = warp_shape_candidates[min_slices_idx]

        # If there is a legal candidate
        else:
            if len(warp_shape_candidates_filtered) > 1:
                effective_block_tiles = [(x[0] * query['block_r'], x[1] * query['block_c'], x[2] * query['head_dim']) for x in warp_shape_candidates_filtered]
                estimated_smem_to_reg = [(effective_block_tiles[i][0] * effective_block_tiles[i][2] * x[1] + effective_block_tiles[i][1] * effective_block_tiles[i][2] * x[0]) for i, x in enumerate(warp_shape_candidates_filtered)]
                minidx = estimated_smem_to_reg.index(min(estimated_smem_to_reg))
            else:
                minidx = 0

            kernel_warp_tile = warp_shape_candidates_filtered[minidx]
            min_slices_needed = 1

        num_warp_tile = (math.ceil(query['block_r'] / kernel_warp_tile[0]), math.ceil(query['block_c'] / kernel_warp_tile[1]), min_slices_needed)
        groupsK = math.ceil(math.ceil(query['head_dim'] / num_warp_tile[2]) / kernel_warp_tile[2])

        parsed['QK_gemm_warp_tile_M'] = kernel_warp_tile[0]
        parsed['QK_gemm_warp_tile_N'] = kernel_warp_tile[1]
        parsed['QK_gemm_warp_tile_K'] = kernel_warp_tile[2]
        parsed['QK_num_gemm_warp_tile_M'] = num_warp_tile[0]
        parsed['QK_num_gemm_warp_tile_N'] = num_warp_tile[1]
        parsed['QK_num_gemm_warp_tile_K'] = num_warp_tile[2]
        parsed['QK_sliceK'] = (num_warp_tile[2] > 1)
        parsed['QK_groupsK'] = groupsK

        # First, assume there is no 'slice-K' (tiling across K-dimension in different warps) and check if there exists a legal one
        num_warps_per_block_candidates = [math.ceil(query['block_r'] / x[0]) * math.ceil(query['head_dim'] / x[1]) for x in warp_shape_candidates]
        warp_shape_candidates_filtered = [warp_shape_candidates[i] for i in range(len(warp_shape_candidates)) if (num_warps_per_block_candidates[i] == parsed['n_warps_per_block'])] #  and (num_inst_per_warp_candidates[i] == tensor_op_inst_per_warp)
        
        # If there is no legal candidate, treat this as 'slice-k'
        if len(warp_shape_candidates_filtered) == 0:
            slicek_candidates_index =[i for i in range(len(warp_shape_candidates)) if (warp_shape_candidates[i][0] <= query['block_r']) and (warp_shape_candidates[i][1] <= query['head_dim'])]
            if len(slicek_candidates_index) == 0:
                slicek_candidates_index = [i for i in range(len(warp_shape_candidates))] # relax
                
            num_slices_needed = [parsed['n_warps_per_block'] / num_warps_per_block_candidates[i] for i in slicek_candidates_index]

            min_slices_needed = min(num_slices_needed)
            min_slices_idx = slicek_candidates_index[num_slices_needed.index(min_slices_needed)]
            kernel_warp_tile = warp_shape_candidates[min_slices_idx]

        # If there is a legal candidate
        else:
            if len(warp_shape_candidates_filtered) > 1:
                effective_block_tiles = [(x[0] * query['block_r'], x[1] * query['head_dim'], x[2] * query['block_c']) for x in warp_shape_candidates_filtered]
                estimated_smem_to_reg = [(effective_block_tiles[i][0] * effective_block_tiles[i][2] * x[1] + effective_block_tiles[i][1] * effective_block_tiles[i][2] * x[0]) for i, x in enumerate(warp_shape_candidates_filtered)]
                minidx = estimated_smem_to_reg.index(min(estimated_smem_to_reg))
            else:
                minidx = 0

            kernel_warp_tile = warp_shape_candidates_filtered[minidx]
            min_slices_needed = 1

        num_warp_tile = (math.ceil(query['block_r'] / kernel_warp_tile[0]), math.ceil(query['head_dim'] / kernel_warp_tile[1]), min_slices_needed)
        groupsK = math.ceil(math.ceil(query['block_c'] / num_warp_tile[2]) / kernel_warp_tile[2])

        parsed['SV_gemm_warp_tile_M'] = kernel_warp_tile[0]
        parsed['SV_gemm_warp_tile_N'] = kernel_warp_tile[1]
        parsed['SV_gemm_warp_tile_K'] = kernel_warp_tile[2]
        parsed['SV_num_gemm_warp_tile_M'] = num_warp_tile[0]
        parsed['SV_num_gemm_warp_tile_N'] = num_warp_tile[1]
        parsed['SV_num_gemm_warp_tile_K'] = num_warp_tile[2]
        parsed['SV_sliceK'] = (num_warp_tile[2] > 1)
        parsed['SV_groupsK'] = groupsK

        parsed['gemm_math_inst_M'] = math_inst_shape_decoded[0]
        parsed['gemm_math_inst_N'] = math_inst_shape_decoded[1]
        parsed['gemm_math_inst_K'] = math_inst_shape_decoded[2]


    def calculate_kernel_info_from_prediction(self, query, kernel_info):
        # provided: block_r, block_c, n_warps_per_block, max_concurrent_block

        for k, v in kernel_info.items():
            query[k] = v

        query['num_block_r'] = int(query['seq_len'] / query['block_r'])
        query['num_block_c'] = int(query['seq_len'] / query['block_c'])
        query['num_block_tile_batch'] = int(query['batch'] * query['n_head'])
        query['total_block_tiles'] = query['num_block_tile_batch'] * query['num_block_r']
        query['iteration_stages'] = query['num_block_c']
        query['multistageK'] = 2
        query['threads'] = 32 * query['n_warps_per_block']

        parsed = {
            'QK_gemm_warp_tile_M': -1, \
            'QK_gemm_warp_tile_N': -1, \
            'QK_gemm_warp_tile_K': -1, \
            'QK_num_gemm_warp_tile_M': -1, \
            'QK_num_gemm_warp_tile_N': -1, \
            'QK_num_gemm_warp_tile_K': -1, \
            'gemm_math_inst_M': -1, \
            'gemm_math_inst_N': -1, \
            'gemm_math_inst_K': -1, \
            'QK_sliceK': -1, \
            'QK_groupsK': -1, \
            'SV_gemm_warp_tile_M': -1, \
            'SV_gemm_warp_tile_N': -1, \
            'SV_gemm_warp_tile_K': -1, \
            'SV_num_gemm_warp_tile_M': -1, \
            'SV_num_gemm_warp_tile_N': -1, \
            'SV_num_gemm_warp_tile_K': -1, \
            'SV_sliceK': -1, \
            'SV_groupsK': -1, \
            'threads': query['threads'], \
            'n_warps_per_block': query['n_warps_per_block']
        }

        self._parse_flashattn_warp(query, parsed, know_threads_warps=True)
        for k,v in parsed.items():
            query[k] = v

        return query

