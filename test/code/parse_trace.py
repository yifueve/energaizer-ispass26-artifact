import os
import numpy as np
import pandas as pd
import yaml
import json
import argparse

from functools import reduce
import operator

from parse_utils import parse_einsum

def get_parent_layer_output_tensor_shape(df, parent_label):
    return eval(df.loc[df['layer_label'] == parent_label, 'tensor_shape'].values[0])

def get_parent_layer_output_tensor_dtype(df, parent_label):
    return df.loc[df['layer_label'] == parent_label, 'tensor_dtype'].values[0]

def get_dtype(dtype):
    if dtype == 'torch.float32':
        return ('fp32', False)
    elif dtype == 'torch.float16':
        return ('fp16', True)
    elif dtype == 'torch.bfloat16':
        return ('bf16', True)
    else:
        raise NotImplementedError()
    
def is_fusible(df, label, fusible_ops):
    op = df.loc[df['layer_label'] == label, 'layer_type'].values[0]
    fusible = (op in fusible_ops)
    if fusible is False:
        return fusible

    childs = eval(df.loc[df['layer_label'] == label, 'child_layers'].values[0])

    # Only enforce a single child
    if len(childs) > 1:
        return False
    
    for c in childs:
        cop = df.loc[df['layer_label'] == label, 'layer_type'].values[0]
        fusible = fusible and (cop in fusible_ops)
        if fusible is False:
            return fusible
    
    return fusible

class fusible_group():
    def __init__(self, endchild):
        self.members = [endchild]
        self.end = endchild
        self.start = []

    def __len__(self):
        return len(self.members)
    
    def has_child(self, node, df):
        childs = eval(df.loc[df['layer_label'] == node, 'child_layers'].values[0])
        for c in childs:
            if c in self.members:
                return True
        return False
    
    def add_member(self, node, is_start=False):
        self.members.append(node)
        if is_start:
            self.start.append(node)

    def prune_member(self):
        self.members = list(set(self.members))
        self.start = list(set(self.start))

    def annotate_trace(self, df):
        for idx, row in df.iterrows():
            if row['layer_label'] in self.members:
                df.loc[idx, 'fusion'] = True
                df.loc[idx, 'fusion_ignore_in'] = (row['layer_label'] not in self.start)
                df.loc[idx, 'fusion_ignore_out'] = (row['layer_label'] != self.end)
                df.loc[idx, 'fused_op_count'] = len(self.members)

def has_fusible_child(df, label):
    childs = eval(df.loc[df['layer_label'] == label, 'child_layers'].values[0])
    child_fusible = False
    for c in childs:
        child_fusible = child_fusible or (df.loc[df['layer_label']==c, 'fusible'].values[0])
        if child_fusible:
            return True
    return False

def has_fusible_parent(df, label):
    parents = eval(df.loc[df['layer_label'] == label, 'parent_layers'].values[0])
    parent_fusible = False
    for c in parents:
        parent_fusible = parent_fusible or (df.loc[df['layer_label']==c, 'fusible'].values[0])
        if parent_fusible:
            return True
    return False

def annotate_fusion(trace, fusible_ops):
    fusible_groups = []
    trace['fusible'] = False
    trace['fusible_not_in_group'] = False

    for idx, row in trace.iterrows():
        fusible = is_fusible(trace, row['layer_label'], fusible_ops)
        trace.loc[idx, 'fusible'] = fusible
    trace.loc[trace['fusible'], 'fusible_not_in_group'] = True

    for idx, row in trace.iterrows():
        if row['fusible']:
            if not has_fusible_child(trace, row['layer_label']):
                g = fusible_group(row['layer_label'])
                fusible_groups.append(g)
                trace.loc[idx, 'fusible_not_in_group'] = False
    
    iter_cnt = 0
    while (trace.loc[trace['fusible'], 'fusible_not_in_group'].any()) and (iter_cnt < 100):
        iter_cnt += 1
        for idx, row in trace.iterrows():
            if row['fusible']:
                for g in fusible_groups:
                    if g.has_child(row['layer_label'], trace):
                        is_start = not has_fusible_parent(trace, row['layer_label'])
                        g.add_member(row['layer_label'], is_start)
                        trace.loc[idx, 'fusible_not_in_group'] = False

    pruned_fusible_groups = []
    for g in fusible_groups:
        if len(g) > 1:
            pruned_fusible_groups.append(g)

    for g in pruned_fusible_groups:
        g.prune_member()
        g.annotate_trace(trace)

def parse_einsum_trace(row, einsum_origin, einsum_cnt_within_origin_module, trace):
    if row['layer_type'] != 'einsum':
        print("This row is not einsum!")
        return -1

    if einsum_origin == 'rope':
        assert(einsum_cnt_within_origin_module == 0) # There should be only one einsum 
        einsum_eq = 'bik,bjk->bij'
        parents = eval(row['parent_layers'])
        parents_shapes = [get_parent_layer_output_tensor_shape(trace, parents[i]) for i in range(len(parents))]
        parents_shapes = [[1, x[0], 1] for x in parents_shapes]
        einsum_args = [einsum_eq] + parents_shapes
    
    elif einsum_origin == 'mlp':
        assert(einsum_cnt_within_origin_module < 3) # There should be at most three einsums
        if einsum_cnt_within_origin_module == 0:
            einsum_eq = 'beck,bk->bec'
        elif einsum_cnt_within_origin_module == 1:
            einsum_eq = 'beck,bek->bec'
        elif einsum_cnt_within_origin_module == 2:
            einsum_eq = 'bec,be->bc'
        
        parents = eval(row['parent_layers'])
        parents_shapes = [list(get_parent_layer_output_tensor_shape(trace, parents[i])) for i in range(len(parents))]
        einsum_args = [einsum_eq] + parents_shapes
    
    elif einsum_origin == 'attn':
        assert(einsum_cnt_within_origin_module < 2) # There should be at most two einsums
        if einsum_cnt_within_origin_module == 0:
            einsum_eq = 'qhmd,khmd->hmqk'
        elif einsum_cnt_within_origin_module == 1:
            einsum_eq = 'hmqk,khmd->qhmd'
        
        parents = eval(row['parent_layers'])
        parents_shapes = [list(get_parent_layer_output_tensor_shape(trace, parents[i])) for i in range(len(parents))]
        einsum_args = [einsum_eq] + parents_shapes

    else:
        raise NotImplementedError("Unrecognized origin type: ", einsum_origin)
    
    parsed = parse_einsum(einsum_args)
    prec, useTensorCore = get_dtype(row['tensor_dtype'])
    queries = []
    for query_vars in parsed:
        # resulttype, resultname, resultvars = torch2cuda_rulebook.resolve('bmm', query_vars)
        resultvars = {}
        resultvars['batch'] = query_vars['matA'][0]
        resultvars['dimM'] = query_vars['matA'][1]
        resultvars['dimN'] = query_vars['matB'][2]
        resultvars['dimK'] = query_vars['matA'][2]
        resultvars['precM'] = prec
        resultvars['precA'] = prec
        resultvars['useTensorCore'] = useTensorCore
        queries.append((resultvars, ('gemm', 'tc' if useTensorCore else 'cuda', '{}_{}'.format(prec, prec))))
    
    return queries

def parse(trace, fusion=False):
    queries = []

    gemm_operation_list = ['linear', 'addmm', 'matmul', 'bmm', 'mm'] # TODO: baddmm
    conv_operation_list = ['conv2d']
    elementwise_operation_list = ['mul', 'add', 'rmul', 'radd', 'div', 'rdiv', 'truediv', 'pow', \
                                  'relu', 'gelu', 'tanh', 'silu', 'sigmoid', 'rsqrt', 'cos', 'sin', \
                                  'sub', 'rsub', 'clamp', 'iadd', 'imul', 'reciprocal']
    elemtnwise_op_map = {'mul': ['pointwise_mul', 'scalar_mul'], \
                         'add': ['pointwise_add', 'scalar_add'], \
                         'rmul': ['pointwise_mul', 'scalar_mul'], \
                         'radd': ['pointwise_mul', 'scalar_mul'], \
                         'rdiv': ['unspecified_tensor', 'unspecified_scalar'], \
                         'div': ['unspecified_tensor', 'unspecified_scalar'], \
                         'truediv': ['unspecified_tensor', 'unspecified_scalar'], \
                         'pow': ['unspecified_tensor', 'unspecified_scalar'], \
                         'rsqrt': ['unspecified_tensor', 'unspecified_scalar'], \
                         'relu': 'relu', 'gelu': 'gelu', 'tanh': 'tanh', 'silu': 'silu', 'sigmoid': 'sigmoid', \
                         'cos': 'unspecified_scalar', 'sin': 'unspecified_scalar', \
                         'sub': ['unspecified_tensor', 'unspecified_scalar'], \
                         'rsub': ['unspecified_tensor', 'unspecified_scalar'], \
                         'clamp': ['unspecified_tensor', 'unspecified_scalar'], \
                         'iadd': ['unspecified_tensor', 'unspecified_scalar'], \
                         'imul': ['unspecified_tensor', 'unspecified_scalar'], \
                         'reciprocal': ['unspecified_tensor', 'unspecified_scalar']}
    
    fusible_operation_list = elementwise_operation_list + ['softmax', 'layernorm']
    trace['fusion'] = False
    trace['fusion_ignore_in'] = False
    trace['fusion_ignore_out'] = False
    trace['fused_op_count'] = 0

    einsum_cnt_dict = {}
    einsum_cnt_dict['rope'] = {}
    einsum_cnt_dict['mlp'] = {}
    einsum_cnt_dict['attn'] = {}

    if fusion:
        annotate_fusion(trace, fusible_operation_list)

    for idx, row in trace.iterrows():
        if row['layer_type'] in gemm_operation_list:

            query = {}

            if type(row['parent_layers']) == str:
                parents = row['parent_layers'][1:-1].replace("'", '').replace(" ", '').split(',')
            else:
                parents = row['parent_layers']
            output_shape = eval(row['tensor_shape']) if type(row['tensor_shape']) == str else row['tensor_shape']
            
            if row['layer_type'] == 'linear':
                assert (row['computed_with_params'] == True)
                assert (len(parents) == 1)
                input_shape = (get_parent_layer_output_tensor_shape(trace, parents[0]))
                weight_shape = (output_shape[-1], input_shape[-1])
                # has_bias = True if (row['num_params_total'] > output_shape[-1] * input_shape[-1]) else False

                query['batch'] = 1
                query['dimM'] = weight_shape[0]                           # Output dimension (M)
                query['dimN'] = reduce(operator.mul, input_shape[:-1], 1) # Batch dimension  (N)
                query['dimK'] = weight_shape[1]                           # Input dimension  (C)
                
            elif (row['layer_type'] == 'bmm') or (row['layer_type'] == 'matmul') or (row['layer_type'] == 'mm'):
                assert (row['computed_with_params'] == False)
                assert (len(parents) == 2)
                matA_shape = (get_parent_layer_output_tensor_shape(trace, parents[0]))
                matB_shape = (get_parent_layer_output_tensor_shape(trace, parents[1]))

                query['batch'] = reduce(operator.mul, matA_shape[:-2], 1)
                query['dimM'] = matA_shape[-2]
                query['dimN'] = matB_shape[-1]
                query['dimK'] = matA_shape[-1]

            elif (row['layer_type'] == 'addmm'):
                assert (row['computed_with_params'] == True) # GPT2 uses addmm for conv1d -> has params
                assert (len(parents) == 1)
                matA_shape = (get_parent_layer_output_tensor_shape(trace, parents[0]))
                matB_shape = eval(row['parent_param_shapes'])[1]

                query['batch'] = 1
                query['dimM'] = matA_shape[-2]
                query['dimN'] = matB_shape[-1]
                query['dimK'] = matA_shape[-1]

            prec, useTensorCore = get_dtype(row['tensor_dtype'])
            query['precM'] = prec
            query['precA'] = prec
            query['useTensorCore'] = useTensorCore
            queries.append((query, ('gemm', 'tc' if useTensorCore else 'cuda', '{}_{}'.format(prec, prec))))

        # PyTorch SDPA backend for attention (assuming flash attention is used)
        if row['layer_type'] == 'scaleddotproductattention':
            query = {}

            if type(row['parent_layers']) == str:
                parents = row['parent_layers'][1:-1].replace("'", '').replace(" ", '').split(',')
            else:
                parents = row['parent_layers']
            output_shape = eval(row['tensor_shape']) if type(row['tensor_shape']) == str else row['tensor_shape']

            assert(len(parents) == 3)
            assert((row['tensor_dtype'] == 'torch.bfloat16') or (row['tensor_dtype'] == 'torch.float16')) # FlashAttention supports FP16/BF16, not FP32

            matQ_shape = get_parent_layer_output_tensor_shape(trace, parents[0])
            matK_shape = get_parent_layer_output_tensor_shape(trace, parents[1])
            matV_shape = get_parent_layer_output_tensor_shape(trace, parents[2])

            q_batch = int(matQ_shape[0])
            q_nhead = int(matQ_shape[1])
            q_seqlen = int(matQ_shape[2])
            q_headdim = int(matQ_shape[3])

            # Check whether grouped/multi-query attention is used
            # Note: for Qwen2 that uses GQA, it seems that K/V are expanded to the full size of Q before fed into sdpa
            # Just in case, compare the total size of q and k tensors to get multi_query_ratio
            k_batch = int(matV_shape[0])
            k_nhead = int(matV_shape[1])
            k_seqlen = int(matV_shape[2])
            k_headdim = int(matV_shape[3])

            if ((q_batch * q_nhead * q_seqlen * q_headdim) != (k_batch * k_nhead * k_seqlen * k_headdim)):
                multi_query_ratio = int((q_batch * q_nhead * q_seqlen * q_headdim) / (k_batch * k_nhead * k_seqlen * k_headdim))
            else:
                multi_query_ratio = 1

            query['batch_size'] = q_batch
            query['num_heads'] = q_nhead
            query['seq_len'] = q_seqlen
            query['head_dim'] = q_headdim
            query['multi_query_ratio'] = multi_query_ratio
            prec, useTensorCore = get_dtype(row['tensor_dtype'])
            query['precM'] = prec
            query['precA'] = prec
            query['useTensorCore'] = useTensorCore
            query['fusion_approx_method'] = 'flash_v2'
            queries.append((query, ('sdpa', )))

        if (row['layer_type'] == 'softmax') or (row['layer_type'] == 'layernorm'):
            output_shape = eval(row['tensor_shape']) if type(row['tensor_shape']) == str else row['tensor_shape']
            prec, _ = get_dtype(row['tensor_dtype'])

            # TODO: Softmax Typecast (OPT, ..)
            softmax_typecast = False
            if row['layer_type'] == 'softmax':
                parents = row['parent_layers']
                if type(parents) == str:
                    parents = parents[1:-1].replace("'", '').replace(" ", '').split(',')

                input_dtype, _ = get_dtype(get_parent_layer_output_tensor_dtype(trace, parents[0]))
                if input_dtype != prec:
                    softmax_typecast = True

                    query = {}
                    query['dim'] = reduce(operator.mul, output_shape, 1)
                    query['op'] = 'typecast_to_fp32'
                    query['prec'] = input_dtype
                    if fusion:
                        query['fusion'] = True
                        query['fusion_ignore_in'] = row['fusion_ignore_in']
                        query['fusion_ignore_out'] = True
                        query['fused_op_count'] = row['fused_op_count']
                    queries.append((query, ('elementwise',)))
                    
                    query = {}
                    query['dim'] = reduce(operator.mul, output_shape, 1)
                    query['op'] = 'typecast_to_bf16'
                    query['prec'] = 'fp32'
                    if fusion:
                        query['fusion'] = True
                        query['fusion_ignore_in'] = True
                        query['fusion_ignore_out'] = row['fusion_ignore_out']
                        query['fused_op_count'] = row['fused_op_count']
                    queries.append((query, ('elementwise',)))

            query = {}
            query['batch'] = reduce(operator.mul, output_shape[:-1], 1)
            query['dim'] = output_shape[-1]
            query['prec'] = prec
            if row['fusion'] or (fusion and softmax_typecast):
                optype = '{}_fusion'.format(row['layer_type'])
                query['fusion_ignore_in'] = row['fusion_ignore_in'] or softmax_typecast
                query['fusion_ignore_out'] = row['fusion_ignore_out'] or softmax_typecast
                query['fusion'] = True
                query['fused_op_count'] = row['fused_op_count']
            else:
                optype = row['layer_type']
            queries.append((query, (optype, prec)))
                

        if row['layer_type'] in conv_operation_list:
            if type(row['parent_layers']) == str:
                parents = row['parent_layers'][1:-1].replace("'", '').replace(" ", '').split(',')
            else:
                parents = row['parent_layers']

            assert (row['computed_with_params'] == True)
            assert (len(parents) == 1)

            input_shape = (get_parent_layer_output_tensor_shape(trace, parents[0]))
            weight_shape = eval(row['parent_param_shapes'])[0]
            output_shape = eval(row['tensor_shape'])

            b = input_shape[0]
            m = output_shape[1]
            c = input_shape[1]
            h = input_shape[2]
            w = input_shape[3]
            p = output_shape[2]
            q = output_shape[3]
            r = weight_shape[2]
            s = weight_shape[3]

            query = {}
            query['batch'] = 1
            query['dimM'] = b * p * q
            query['dimN'] = m
            query['dimK'] = c * r * s
            prec, _ = get_dtype(row['tensor_dtype'])
            query['precM'] = prec
            query['precA'] = prec
            query['useTensorCore'] = True
            
            query['b'] = b
            query['m'] = m
            query['c'] = c
            query['hw'] = h
            query['rs'] = r
            query['stride'] = row['conv_stride']
            query['padding'] = row['conv_padding']
            queries.append((query, ('conv2d', 'tc', '{}_{}'.format(prec, prec))))

        if row['layer_type'] in elementwise_operation_list:
            op_mapped = elemtnwise_op_map[row['layer_type']]
            if type(row['parent_layers']) == str:
                parents = row['parent_layers'][1:-1].replace("'", '').replace(" ", '').split(',')
            else:
                parents = row['parent_layers']

            if len(parents) == 1:
                if type(op_mapped) == list:
                    op_mapped = op_mapped[1]
            elif len(parents) == 2:
                if type(op_mapped) == list:
                    op_mapped = op_mapped[0]
            else:
                raise ValueError("There are more than 2 parents for this layer!", row)
            
            parent_shapes = []
            for p in parents:
                input_shape = (get_parent_layer_output_tensor_shape(trace, p))
                parent_shapes.append(input_shape)

            if len(parents) == 2:
                parents0_dim = reduce(operator.mul, parent_shapes[0], 1) if len(parent_shapes[0]) > 0 else 0
                parents1_dim = reduce(operator.mul, parent_shapes[1], 1) if len(parent_shapes[1]) > 0 else 0

                # print(parents[0], parents0_dim, parents[1], parents1_dim)

                if parents0_dim  > parents1_dim:
                    parent_shape = parent_shapes[0]
                else:
                    parent_shape = parent_shapes[1]
            else:
                parent_shape = parent_shapes[0]

            try:
                prec, _ = get_dtype(row['tensor_dtype'])
                query = {}
                query['dim'] = reduce(operator.mul, parent_shape, 1)
                query['op'] = op_mapped
                query['prec'] = prec
                if fusion:
                    query['fusion'] = row['fusion']
                    query['fusion_ignore_in'] = row['fusion_ignore_in']
                    query['fusion_ignore_out'] = row['fusion_ignore_out']
                    query['fused_op_count'] = row['fused_op_count']

                queries.append((query, ('elementwise',)))

            except Exception as e:
                # print(e)
                # print(row['tensor_dtype'])
                pass

        if row['layer_type'] == 'einsum':
            # Three origin module type: rope, mlp, attn
            if 'rope' in row['containing_module_origin']:
                origin_type = 'rope'
            elif 'mlp' in row['containing_module_origin']:
                origin_type = 'mlp'
            elif 'attn' in row['containing_module_origin']:
                origin_type = 'attn'
            else:
                raise NotImplementedError("Unrecognized origin type: ", row['containing_module_origin'])
            
            if row['containing_module_origin'] in einsum_cnt_dict[origin_type].keys():
                cnt = einsum_cnt_dict[origin_type][row['containing_module_origin']]
            else:
                cnt = 0
                einsum_cnt_dict[origin_type][row['containing_module_origin']] = 0
            
            einsum_queries = parse_einsum_trace(row, origin_type, cnt, trace)
            einsum_cnt_dict[origin_type][row['containing_module_origin']] += 1
            for eq in einsum_queries:
                queries.append(eq)

    return queries

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--trace_path', default='tmp/trace/', type=str)
    parser.add_argument('--parsed_save_to', default='tmp/trace_parsed/', type=str)
    parser.add_argument('--fusion', default=False, action='store_true')

    args = parser.parse_args()
    
    if not os.path.exists(args.parsed_save_to):
        os.mkdir(args.parsed_save_to)

    for filename in os.listdir(args.trace_path):
        if filename.endswith('csv'):
            full_path = os.path.join(args.trace_path, filename)
            json_filename = filename.replace('csv', 'json')
            json_filepath = os.path.join(args.parsed_save_to, json_filename)

            trace = pd.read_csv(full_path)
            queries = parse(trace, args.fusion)
            with open(json_filepath, 'w') as f:
                json.dump(queries, f)

if __name__ == '__main__':
    main()
