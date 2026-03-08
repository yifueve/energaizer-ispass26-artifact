# Frontend utility functions for GEE
import os
import numpy as np
import pandas as pd
import math
import copy
import csv
import json
import re
import yaml

import warnings
warnings.filterwarnings('ignore')

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx

# optimized einsum
from opt_einsum.parser import parse_einsum_input

## Einsum parsing
def parse_einsum(einsum_args, einsum_parse_cache, enable_einsum_cache, compute_output=False):
    # First, check if the equation has been encountered before.
    # If so, get lro_size, lo_size, ro_size, sum_size to generate the gemm workload
    # is_one computation
    query_is_one = []
    query_operands = list(einsum_args[1:])
    for op in query_operands:
        if type(op) == torch.Tensor:
            sizes = list(op.shape)
        elif type(op) == list:
            sizes = op
        else:
            raise TypeError('Einsum Argument Operands should be either torch.Tensor or python list')
        
        _is_one = [x == 1 for x in sizes]
        query_is_one.append(_is_one)

    if (einsum_args[0]) in einsum_parse_cache.keys():
        entries = einsum_parse_cache[einsum_args[0]]
        cache_hit = False
        op_rule = None
        for e in entries:
            if e[1] == query_is_one:
                cache_hit = True
                op_rule = e[0]
                break
        if cache_hit and (op_rule is not None):
            # print("Cache hit!")
            gemm_list = []
            # op_rule = einsum_parse_cache[einsum_args[0]][0]
            ops = einsum_args[1:]
            for idx in range(1, len(ops)):
                if idx == 1:
                    ruleA = op_rule[idx - 1]
                    ruleB = op_rule[idx]
                    opAshape = ops[idx - 1].shape if type(ops[idx-1]) == torch.Tensor else ops[idx-1]
                    opBshape = ops[idx].shape if type(ops[idx]) == torch.Tensor else ops[idx]

                    lro_size = 1
                    lo_size = 1
                    ro_size = 1
                    sum_size = 1

                    for i in ruleA[0]:
                        if i == -1:
                            continue
                        lro_size *= opAshape[i]
                    for i in ruleA[1]:
                        if i == -1:
                            continue
                        lo_size *= opAshape[i]
                    for i in ruleA[2]:
                        if i == -1:
                            continue
                        sum_size *= opAshape[i]
                    for i in ruleB[2]:
                        if i == -1:
                            continue
                        ro_size *= opBshape[i]

                    # gemm_args = {'matA': torch.rand(lro_size, lo_size, sum_size), 'matB': torch.rand(lro_size, sum_size, ro_size), 'result': torch.rand(lro_size, lo_size, ro_size)}
                

                else:
                    ruleB = op_rule[idx]
                    opBshape = ops[idx].shape if type(ops[idx]) == torch.Tensor else ops[idx]

                    lro_size = 1
                    # lo_size = 1 --> previous layer's
                    ro_size = 1
                    sum_size = 1

                    for i in ruleB[0]:
                        if i == -1:
                            continue
                        lro_size *= opBshape[i]
                    for i in ruleB[1]:
                        if i == -1:
                            continue
                        sum_size *= opBshape[i]
                    for i in ruleA[2]:
                        if i == -1:
                            continue
                        ro_size *= opBshape[i]

                gemm_args = {'matA': (lro_size, lo_size, sum_size), 'matB': (lro_size, sum_size, ro_size), 'result': (lro_size, lo_size, ro_size)}

                gemm_list.append(gemm_args)

            return gemm_list
        
    # For cache miss, we have to get actual torch.Tensor
    original_einsum_args = copy.deepcopy(einsum_args)
    ops = list(einsum_args[1:])
    for i, op in enumerate(ops):
        if type(op) != torch.Tensor:
            if enable_einsum_cache:
                smaller_dim = [2 if x > 1 else 1 for x in op]
                ops[i] = torch.zeros(smaller_dim)
            else:
                ops[i] = torch.zeros(op)
    einsum_args = tuple([einsum_args[0]] + ops)

    # einsum_args = (equation, operand1, operand2, ...)
    input_subscripts, output_subscript, operands = parse_einsum_input(einsum_args)
    # input_indices = list(set(list(input_subscripts.replace(',', ''))))
    output_indices = list(set(list(output_subscript)))
    # contract_indices = list(set(input_indices) - set(output_indices))

    op_manipulation_rule = []

    num_of_letters = ord('z') - ord('a') + 1
    total_labels = num_of_letters * 2

    def char_to_int(x, num_of_letters):
        if x.isupper():
            return ord(x) - ord('A')
        else:
            return ord(x) - ord('a') + num_of_letters
        
    def int_to_char(x, num_of_letters):
        if x < num_of_letters:
            return chr(x + ord('A'))
        else:
            return chr(x + ord('a') - num_of_letters)

    label_perm_index = {}
    for idx in range(total_labels):
        label_perm_index[idx] = -1

    perm_index = 0

    # output subscripts
    for idx in range(len(output_subscript)):
        label_perm_index[char_to_int(output_subscript[idx], num_of_letters)] = perm_index
        perm_index += 1

    # contract dimensions in alphabetical order
    for idx in range(total_labels):
        a = int_to_char(idx, num_of_letters)
        if (a in input_subscripts) and (label_perm_index[idx] == -1):
            label_perm_index[idx] = perm_index
            perm_index += 1

    label_size = {}
    for idx in range(total_labels):
        label_size[idx] = 1

    dim_counts = {}
    for idx in range(perm_index):
        dim_counts[idx] = 0

    operands = list(einsum_args)[1:]
    op_labels = input_subscripts.split(',')
    new_operands = []

    # for each operand
    gemm_list = []
    for op_idx, op in enumerate(operands):
        rule = list(range(op.dim()))
        permutation = {}
        for idx in range(perm_index):
            permutation[idx] = -1
        dim = 0
        label = op_labels[op_idx]
        for s in label:
            if (permutation[label_perm_index[char_to_int(s, num_of_letters)]] == -1):
                if (op.shape[dim] != 1):
                    assert ((label_size[char_to_int(s, num_of_letters)] == 1) or (label_size[char_to_int(s, num_of_letters)] == op.shape[dim]))
                    label_size[char_to_int(s, num_of_letters)] = op.shape[dim]
                    dim_counts[label_perm_index[char_to_int(s, num_of_letters)]] += 1
                permutation[label_perm_index[char_to_int(s, num_of_letters)]] = dim
                dim += 1
            else:
                # repeated label, take diagonal
                prev_dim = permutation[label_perm_index[char_to_int(s, num_of_letters)]]
                assert (op.shape[dim] == op.shape[prev_dim])
                op = op.diagonal(0, prev_dim, dim).movedim(-1, prev_dim)
                rule.remove(dim)
            
        # missing labels -> unsqueeze
        for key in permutation.keys():
            if permutation[key] == -1:
                op = op.unsqueeze(dim)
                permutation[key] = dim
                rule.append(-1)
                dim += 1
        
        permutation = list(permutation.values())
        new_operands.append(op.permute(permutation))

        _rule = [rule[x] for x in permutation]
        op_manipulation_rule.append(_rule)

    curr_result = None
    op_final_rule = []
    # print(len(new_operands))
    for idx in range(1, len(new_operands)):
        if idx == 1:
            opA = new_operands[idx - 1]
            opB = new_operands[idx]
            ruleA = op_manipulation_rule[idx - 1]
            ruleB = op_manipulation_rule[idx]
        else:
            opA = curr_result
            opB = new_operands[idx]
            ruleA = list(range(opA.dim()))
            ruleB = op_manipulation_rule[idx]

        # print(opA.shape, opB.shape)
        
        sum_dims = []
        a_dims_to_sum = []
        b_dims_to_sum = []

        for dim in range(len(output_indices), perm_index):
            if (opA.shape[dim] != 1) and (opB.shape[dim] != 1):
                if (dim_counts[dim] - 1 == 1):
                    sum_dims.append(dim)
                    dim_counts[dim] = 0
            elif (dim_counts[dim] == 1):
                if (opA.shape[dim] != 1):
                    a_dims_to_sum.append(dim)
                    dim_counts[dim] = 0
                    ruleA[dim] = -1
                elif (opB.shape[dim] != 1):
                    b_dims_to_sum.append(dim)
                    dim_counts[dim] = 0
                    ruleB[dim] = -1
            
        if (len(a_dims_to_sum) > 0):
            opA = opA.sum(a_dims_to_sum, True)
        if (len(b_dims_to_sum) > 0):
            opB = opB.sum(b_dims_to_sum, True)
        
        # sum-product pair for opA and opB
        assert (opA.dim() == opB.dim())

        if len(sum_dims) == 0:
            # broadcast + hadamard product (not GEMM)
            if compute_output:
                curr_result = torch.mul(opA, opB)
            else:
                _out_size = []
                for i in range(opA.dim()):
                    if (opA.shape[i] == 1) and (opB.shape[i] == 1):
                        _out_size.append(1)
                    elif (opA.shape[i] == 1):
                        _out_size.append(opB.shape[i])
                    else:
                        _out_size.append(opA.shape[i])
                curr_result = torch.rand(_out_size)
            # instead of actually computing the result
            # break

        _dim = opA.dim()
        sum_size = 1
        lro_size = 1
        lo_size = 1
        ro_size = 1
        lro = []
        lo = []
        ro = []
        for i in range(_dim):
            sl = opA.shape[i] != 1
            sr = opB.shape[i] != 1
            if (i in sum_dims):
                if (sl and sr):
                    sum_size *= opA.shape[i]
                    assert (opA.shape[i] == opB.shape[i])
                elif (sl):
                    opA = opA.sum(i, True)
                    ruleA[i] = -1
                elif (sr):
                    opB = opB.sum(i, True)
                    ruleB[i] = -1
            elif (sl and sr):
                assert (opA.shape[i] == opB.shape[i])
                lro.append(i)
                lro_size *= opA.shape[i]
            elif (sl):
                lo.append(i)
                lo_size *= opA.shape[i]
            else:
                ro.append(i)
                ro_size *= opB.shape[i]

        # print(lro, lo, ro, sum_dims)
        
        out_num_dim = len(lro) + len(lo) + len(ro) + len(sum_dims)
        out_size = []
        for i in lro:
            out_size.append(opA.shape[i])
        for i in lo:
            out_size.append(opA.shape[i])
        for i in sum_dims:
            out_size.append(1)
        for i in ro:
            out_size.append(opB.shape[i])
        
        lpermutation = lro + lo + sum_dims + ro
        rpermutation = lro + sum_dims + ro + lo
        opermutation = [-1] * out_num_dim
        i = 0
        for it in (lro):
            opermutation[it] = i
            i += 1
        for it in (lo):
            opermutation[it] = i
            i += 1
        for it in (sum_dims):
            # print(it, i)
            opermutation[it] = i
            i += 1
        for it in (ro):
            # print(it, i)
            opermutation[it] = i
            i += 1

        # print(opermutation)
        _ruleA = ([ruleA[i] for i in lro] if len(lro) > 0 else [], [ruleA[i] for i in lo] if len(lo) > 0 else [], [ruleA[i] for i in sum_dims] if len(sum_dims) > 0 else [])
        _ruleB = ([ruleB[i] for i in lro] if len(lro) > 0 else [], [ruleB[i] for i in sum_dims] if len(sum_dims) > 0 else [], [ruleB[i] for i in ro] if len(ro) > 0 else [])

        if idx == 1:
            op_final_rule.append(_ruleA)
            op_final_rule.append(_ruleB)
        else:
            op_final_rule.append(_ruleB)

        opA = opA.permute(lpermutation).view((lro_size, lo_size, sum_size))
        opB = opB.permute(rpermutation).view((lro_size, sum_size, ro_size))

        # print(opA.shape, opB.shape)
        if compute_output:
            curr_result = torch.bmm(opA, opB).view(out_size).permute(opermutation)
        else:
            curr_result = torch.rand(out_size).permute(opermutation)

        gemm_args = {'matA': (opA.shape[0], opA.shape[1], opA.shape[2]), 'matB': (opB.shape[0], opB.shape[1], opB.shape[2]), 'result': (curr_result.shape[0], curr_result.shape[1], curr_result.shape[2])}
        gemm_list.append(gemm_args)
        # print(curr_result.shape)

    if compute_output:
        print("Computed output shape", curr_result.squeeze(list(range(len(output_indices), perm_index))).shape)

    if enable_einsum_cache:
        # Determine if any of the dimensions are 0
        operands = list(einsum_args)[1:]
        is_one = []
        for op in operands:
            sizes = list(op.shape)
            _is_one = [x == 1 for x in sizes]
            is_one.append(_is_one)
        if einsum_args[0] in einsum_parse_cache.keys():
            einsum_parse_cache[einsum_args[0]].append((op_final_rule, is_one))
        else:
            einsum_parse_cache[einsum_args[0]] = [(op_final_rule, is_one)]

    if enable_einsum_cache:
        return parse_einsum(original_einsum_args, einsum_parse_cache, enable_einsum_cache)
    else:
        return gemm_list

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
    
def fit_vector_size(batch, m, n, k, max_entries=2**30):
    if compute_vector_size(batch, m, n, k) > max_entries:
        if batch > 2:
            batch = int(batch/2)
        else:
            if max(m, n, k) == m:
                m = int(m/2)
            elif max(m, n, k) == n:
                n = int(n/2)
            else:
                k = int(k/2)
        return fit_vector_size(batch, m, n, k, max_entries)
    else:
        return batch, m, n, k