import os
import numpy as np
import pandas as pd
import json
import argparse
import time
import math
import gc

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn.attention import SDPBackend, sdpa_kernel

# Huggingface Transformers
import transformers
from transformers.cache_utils import Cache, DynamicCache

# HF OPT
from transformers.models.opt.modeling_opt import OPTDecoderLayer, OPTModel
from transformers.models.opt.configuration_opt import OPTConfig

# HF BERT
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.configuration_bert import BertConfig

# HF GPT2
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

# HF QWEN2
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

# HF ViT
from transformers.models.vit.modeling_vit import ViTModel
from transformers.models.vit.configuration_vit import ViTConfig

# HF MobileViT
from transformers.models.mobilevit.modeling_mobilevit import MobileViTModel
from transformers.models.mobilevit.configuration_mobilevit import MobileViTConfig

# HF Auto
from transformers import AutoConfig, AutoModelForCausalLM

# Torchvision Models
import torchvision.models as models

# NVML
try:
    import pynvml
except:
    pass

# TorchLens
import torchlens as tl

os.environ["MKL_THREADING_LAYER"] = 'GNU'

# https://github.com/ml-energy/zeus/blob/master/zeus/monitor/power.py#L47
#### NVML polling function ####
def poll_nvml(gpu_list, save_to, update_period=0.01, poll_clock=False):
    try:
        pynvml.nvmlInit()
        with open(save_to, "a", buffering=1) as f:
            while True:
                stats = []
                for rank in gpu_list:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(rank)
                    # power
                    metric = pynvml.nvmlDeviceGetFieldValues(handle, [pynvml.NVML_FI_DEV_POWER_INSTANT])[0]
                    stats.append(metric.value.uiVal)
                    # temp
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    stats.append(temp)
                    # sm clock freq
                    if poll_clock:
                        freq = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                        stats.append(freq)
                    
                # stats = pynvml.nvmlDeviceGetPowerUsage(nvml_handle)
                stats_str = ",".join(map(lambda p: str(p / 1000), stats))
                now = time.time()
                f.write(f"{now},{stats_str}\n")
                if (sleep_time := update_period - (time.time() - now)) > 0:
                    time.sleep(sleep_time)
    except KeyboardInterrupt:
        return

def get_time_per_iter(module, model_type, input_t, past_kv, use_cache, original_seq_len):
    if (model_type[0] == 'Module') and (model_type[1] == 'OPTDecoder'):
        with torch.no_grad():
            for i in range(10):
                if use_cache:
                    past_kv.key_cache[0] = past_kv.key_cache[0][:, :, :original_seq_len, :]
                    past_kv.value_cache[0] = past_kv.value_cache[0][:, :, :original_seq_len, :]
                _ = module(input_t, past_key_value=past_kv, use_cache=use_cache)
                torch.cuda.synchronize()
        
        start_time = time.time()
        with torch.no_grad():
            if use_cache:
                past_kv.key_cache[0] = past_kv.key_cache[0][:, :, :original_seq_len, :]
                past_kv.value_cache[0] = past_kv.value_cache[0][:, :, :original_seq_len, :]
            _ = module(input_t, past_key_value=past_kv, use_cache=use_cache)
            torch.cuda.synchronize()
        end_time = time.time()

    elif model_type[0] == 'LanguageModel':
        with torch.no_grad():
            for i in range(10):
                if use_cache:
                    if type(past_kv) == list:
                        for kv in past_kv:
                            kv[0] = kv[0][:, :, :original_seq_len, :]
                            kv[1] = kv[1][:, :, :original_seq_len, :]
                    else:
                        for j, k in enumerate(past_kv.key_cache):
                            past_kv.key_cache[j] = k[:, :, :original_seq_len, :]
                        for j, v in enumerate(past_kv.value_cache):
                            past_kv.value_cache[j] = v[:, :, :original_seq_len, :]
                _ = module(input_t, past_key_values=past_kv, use_cache=use_cache)
                torch.cuda.synchronize()
        
        start_time = time.time()
        with torch.no_grad():
            if use_cache:
                if type(past_kv) == list:
                        for kv in past_kv:
                            kv[0] = kv[0][:, :, :original_seq_len, :]
                            kv[1] = kv[1][:, :, :original_seq_len, :]
                else:
                    for j, k in enumerate(past_kv.key_cache):
                        past_kv.key_cache[j] = k[:, :, :original_seq_len, :]
                    for j, v in enumerate(past_kv.value_cache):
                        past_kv.value_cache[j] = v[:, :, :original_seq_len, :]
            _ = module(input_t, past_key_values=past_kv, use_cache=use_cache)
            torch.cuda.synchronize()
        end_time = time.time()

    elif model_type[0] == 'VisionModel':
        with torch.no_grad():
            for i in range(10):
                _ = module(input_t)
                torch.cuda.synchronize()
        
        start_time = time.time()
        with torch.no_grad():
            _ = module(input_t)
            torch.cuda.synchronize()
        end_time = time.time()

    else:
        raise NotImplementedError()
    
    return (end_time - start_time)

def run_iterations(module, model_type, input_t, past_kv, use_cache, repeats, original_seq_len):
    if (model_type[0] == 'Module') and (model_type[1] == 'OPTDecoder'):
        with torch.no_grad():
            for i in range(repeats):
                if use_cache:
                    past_kv.key_cache[0] = past_kv.key_cache[0][:, :, :original_seq_len, :]
                    past_kv.value_cache[0] = past_kv.value_cache[0][:, :, :original_seq_len, :]
                _ = module(input_t, past_key_value=past_kv, use_cache=use_cache)
                torch.cuda.synchronize()

    elif model_type[0] == 'LanguageModel':
        with torch.no_grad():
            for i in range(repeats):
                if use_cache:
                    if type(past_kv) == list:
                        for kv in past_kv:
                            kv[0] = kv[0][:, :, :original_seq_len, :]
                            kv[1] = kv[1][:, :, :original_seq_len, :]
                    else:
                        for j, k in enumerate(past_kv.key_cache):
                            past_kv.key_cache[j] = k[:, :, :original_seq_len, :]
                        for j, v in enumerate(past_kv.value_cache):
                            past_kv.value_cache[j] = v[:, :, :original_seq_len, :]
                _ = module(input_t, past_key_values=past_kv, use_cache=use_cache)
                torch.cuda.synchronize()

    elif model_type[0] == 'VisionModel':
        with torch.no_grad():
            for i in range(repeats):
                _ = module(input_t)
                torch.cuda.synchronize()

    else:
        raise NotImplementedError()

def single_run(module, model_type, input_t, past_kv, use_cache):
    if (model_type[0] == 'Module') and (model_type[1] == 'OPTDecoder'):
        with torch.no_grad():
            _ = module(input_t, past_key_value=past_kv, use_cache=use_cache)

    elif model_type[0] == 'LanguageModel':
        with torch.no_grad():
            _ = module(input_t, past_key_values=past_kv, use_cache=use_cache)

    elif model_type[0] == 'VisionModel':
        with torch.no_grad():
            _ = module(input_t)

    else:
        raise NotImplementedError()
    
def get_model(model_type, config_file_path, device, dtype, attn_backend, hf_token=None):
    
    # Load configuration (for HF models)
    if config_file_path is not None:
        with open(config_file_path, 'r') as f:
            hf_config = json.load(f)

    def change_config(config, hf_config):
        for key, value in hf_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
        if hasattr(config, '_attn_implementation'):
            config._attn_implementation = attn_backend

    config = None

    if (model_type[0] == 'Module') and (model_type[1] == 'OPTDecoder'):
        config = OPTConfig()
        change_config(config, hf_config)
        module = OPTDecoderLayer(config, layer_idx=0)
        
    elif (model_type[0] == 'LanguageModel'):
        if (model_type[1] == 'OPTModel'):
            config = OPTConfig()
            change_config(config, hf_config)
            module = OPTModel(config)

        elif (model_type[1] == 'BERTModel'):
            config = BertConfig()
            change_config(config, hf_config)
            module = BertModel(config)
        
        elif (model_type[1] == 'GPT2Model'):
            config = GPT2Config()
            change_config(config, hf_config)
            module = GPT2Model(config)

        elif (model_type[1] == 'Qwen2Model'):
            config = Qwen2Config()
            change_config(config, hf_config)
            module = Qwen2Model(config)

        elif (model_type[1] == 'Llama3Model'):
            config = AutoConfig.from_pretrained("meta-llama/Meta-Llama-3-8B", token=hf_token)
            module = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
        
        else:
            raise NotImplementedError()
        
    elif (model_type[0] == 'VisionModel'):
        if (model_type[1] == 'ViTModel'):
            config = ViTConfig()
            change_config(config, hf_config)
            module = ViTModel(config)
        
        elif (model_type[1] == 'MobileViTModel'):
            config = MobileViTConfig()
            change_config(config, hf_config)
            module = MobileViTModel(config)

        elif (model_type[1] == 'ResNet18'):
            module = models.resnet18(pretrained=False)
        
        elif (model_type[1] == 'ResNet34'):
            module = models.resnet34(pretrained=False)
        
        elif (model_type[1] == 'ResNet50'):
            module = models.resnet50(pretrained=False)

        elif (model_type[1] == 'ResNet101'):
            module = models.resnet101(pretrained=False)

        else:
            raise NotImplementedError()
            
    module.to(device=device, dtype=dtype)
    module.eval()
    return module, config

def get_input(model_type, config, batch, seqlen, mode, dtype, device):
    if (model_type[0] == 'Module') and (model_type[1] == 'OPTDecoder'):
        
        if mode == 'prefill':
            input_t = torch.rand(batch, seqlen, config.hidden_size, device=device, dtype=dtype)
            past_kv = None
            use_cache = False
        else:
            input_t = torch.rand(batch, 1, config.hidden_size, device=device, dtype=dtype)
            past_k = torch.rand(batch, config.num_attention_heads, seqlen, int(config.hidden_size / config.num_attention_heads), device=device, dtype=dtype)
            past_v = torch.rand(batch, config.num_attention_heads, seqlen, int(config.hidden_size / config.num_attention_heads), device=device, dtype=dtype)
            past_kv = [[past_k, past_v]]
            past_kv = DynamicCache.from_legacy_cache(past_kv)
            use_cache = True

    elif (model_type[0] == 'LanguageModel'):

        if mode == 'prefill':
            input_t = torch.randint(0, config.vocab_size, (batch, seqlen)).long().to(device=device)
            past_kv = None
            use_cache = False
        else:
            input_t = torch.randint(0, config.vocab_size, (batch, 1)).long().to(device=device)
            past_kv = []
            for i in range(config.num_hidden_layers):
                k = torch.rand(batch, config.num_attention_heads, seqlen, int(config.hidden_size / config.num_attention_heads), device=device, dtype=dtype)
                v = torch.rand(batch, config.num_attention_heads, seqlen, int(config.hidden_size / config.num_attention_heads), device=device, dtype=dtype)
                past_kv.append([k, v])
            use_cache = True

    elif (model_type[0] == 'VisionModel'):
        if config is not None:
            input_t = torch.rand(batch, 3, config.image_size, config.image_size, dtype=dtype, device=device)
        else:
            input_t = torch.rand(batch, 3, 224, 224, dtype=dtype, device=device)
        
        past_kv = None
        use_cache = False

    return {'input': input_t, 'past_kv': past_kv, 'use_cache': use_cache}

def run_torchlens(module, model_type, input_t, past_kv, use_cache, original_seq_len):
    if (model_type[0] == 'Module') and (model_type[1] == 'OPTDecoder'):
        module.eval()
        if use_cache:
            past_kv.key_cache[0] = past_kv.key_cache[0][:, :, :original_seq_len, :]
            past_kv.value_cache[0] = past_kv.value_cache[0][:, :, :original_seq_len, :]
        model_history = tl.log_forward_pass(module, input_t, input_kwargs={'past_key_value': past_kv, 'use_cache': use_cache})
            
    elif model_type[0] == 'LanguageModel':
        module.eval()
        if use_cache:
            if type(past_kv) == list:
                for kv in past_kv:
                    kv[0] = kv[0][:, :, :original_seq_len, :]
                    kv[1] = kv[1][:, :, :original_seq_len, :]
            else:
                for j, k in enumerate(past_kv.key_cache):
                    past_kv.key_cache[j] = k[:, :, :original_seq_len, :]
                for j, v in enumerate(past_kv.value_cache):
                    past_kv.value_cache[j] = v[:, :, :original_seq_len, :]
        
        try:
            model_history = tl.log_forward_pass(module, input_t, input_kwargs={'past_key_values': past_kv, 'use_cache': use_cache})
        except:
            model_history = tl.log_forward_pass(module, None, input_kwargs={'input_ids': input_t, 'past_key_values': past_kv, 'use_cache': use_cache})

    elif model_type[0] == 'VisionModel':
        module.eval()
        model_history = tl.log_forward_pass(module, input_t)

    else:
        raise NotImplementedError()
    
    df = model_history.to_pandas()

    # For vision models with convolution, annotate stride/padding info for conv layers
    if model_type[0] == 'VisionModel':
        if 'conv2d' in df['layer_type'].unique().tolist():
            
            conv_layer_info = {}
            for name, m in module.named_modules():
                if isinstance(m, nn.Conv2d):
                    conv_layer_info[name] = {'stride': m.stride[0], 'padding': m.padding[0]}
        
            df['conv_stride'] = -1
            df['conv_padding'] = -1

            for idx, row in df.iterrows():
                if row['layer_type'] == 'conv2d':
                    name_in_module = row['modules_exited'][0]
                    df.loc[idx, 'conv_stride'] = conv_layer_info[name_in_module]['stride']
                    df.loc[idx, 'conv_padding'] = conv_layer_info[name_in_module]['padding']

    return df

def main():
    parser = argparse.ArgumentParser()

    # Sudo Password if needed
    parser.add_argument('--sudo_pwd', default='', type=str)

    # GPU Setting
    parser.add_argument('--cuda_device', default=0, type=int)
    parser.add_argument('--lock_gpu_clock', default=False, action='store_true')
    parser.add_argument('--gpu_clock_freq', default=-1, type=int)

    # NVML
    parser.add_argument('--run_nvml', default=False, action='store_true')
    parser.add_argument('--nvml_save_to', default='nvml.csv', type=str)
    parser.add_argument('--nvml_update_period', type=float, default=0.01, help='nvml update interval in s')
    parser.add_argument('--nvml_poll_clock', default=False, action='store_true')

    # NCU
    parser.add_argument('--ncu', default=False, action='store_true')

    # Workload Model
    parser.add_argument('--model_type', type=str, choices=['Module', 'LanguageModel', 'VisionModel'], required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--config_file', default=None)
    parser.add_argument('--precision', type=str, choices=['bf16', 'fp16', 'fp32'])

    # Workload Input
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--seqlen', type=int, default=1, help='Sequence length for language models')
    parser.add_argument('--mode', type=str, choices=['prefill', 'decode'], help='Mode (prefill or decode) for language models')

    # Compile?
    parser.add_argument('--compile', default=False, action='store_true')
    parser.add_argument('--attn_backend', default='eager', choices=['eager', 'sdpa'], type=str)

    # TorchLens?
    parser.add_argument('--trace', default=False, action='store_true')
    parser.add_argument('--trace_save_to', default='traced.csv', type=str)

    # Report time per run?
    parser.add_argument('--print_time_per_iter', default=False, action='store_true')

    # HF gated model tokens (e.g., Llama3)
    parser.add_argument('--hf_token', type=str, help='Your token for Huggingface gated models')

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.precision == 'bf16':
        dtype = torch.bfloat16
    elif args.precision == 'fp16':
        dtype = torch.float16
    else:
        dtype = torch.float32

    module, config = get_model((args.model_type, args.model), args.config_file, device, dtype, args.attn_backend, hf_token=args.hf_token)
    input_dict = get_input((args.model_type, args.model), config, args.batch, args.seqlen, args.mode, dtype, device)

    if args.compile:
        torch._dynamo.reset()
        module = torch.compile(module)

    # GPU Frequency locking if specified
    if args.lock_gpu_clock:
        os.system('echo {} | sudo -S nvidia-smi -i {} -lgc {},{}'.format(args.sudo_pwd, args.cuda_device, args.gpu_clock_freq, args.gpu_clock_freq))

    try:
        if args.ncu:
            single_run(module, (args.model_type, args.model), input_t=input_dict['input'], past_kv=input_dict['past_kv'], use_cache=input_dict['use_cache'])

        else:
            time_per_iter = get_time_per_iter(module, (args.model_type, args.model), input_t=input_dict['input'], past_kv=input_dict['past_kv'], use_cache=input_dict['use_cache'], original_seq_len=args.seqlen)
            if args.print_time_per_iter:
                print("Time per iteration: {:4f} ms".format(time_per_iter * 1000.))
            num_iter = math.ceil(30.0 / time_per_iter)

            # Trace
            if args.trace:
                df_trace = run_torchlens(module, (args.model_type, args.model), input_t=input_dict['input'], past_kv=input_dict['past_kv'], use_cache=input_dict['use_cache'], original_seq_len=args.seqlen)
                df_trace.to_csv(args.trace_save_to, index=False)

            # NVML
            if args.run_nvml:
                nvml_save_to = args.nvml_save_to.replace('.csv', '')
                nvml_save_to += '_iter{}.csv'.format(num_iter)

                # Single GPU
                size = 1
                pynvml.nvmlInit()
                with open(nvml_save_to, 'w') as f:
                    header = ['timestamp']
                    for rank in range(size):
                        header.append('power_{}'.format(rank))
                        header.append('temp_{}'.format(rank))
                        if args.nvml_poll_clock:
                            header.append('sm_clock_{}'.format(rank))
                    header_str = ','.join(str(x) for x in header)
                    header_str += '\n'
                    f.write(header_str)
                    # print(header_str)
                    f.close() 

                # mp.set_start_method("spawn")

                # p1 is NVML
                p1 = mp.Process(target=poll_nvml, args=([args.cuda_device], nvml_save_to, args.nvml_update_period, args.nvml_poll_clock,))
                p1.start()

                # p2 is torch
                p2 = mp.Process(target=run_iterations, args=(module, (args.model_type, args.model), input_dict['input'], input_dict['past_kv'], input_dict['use_cache'], num_iter, args.seqlen))
                p2.start()
                
                p2.join()
                p1.terminate()
                p1.join()

                pynvml.nvmlShutdown()


    except Exception as e:
        print(e)

        module = None
        input_dict = None
        gc.collect()
        torch.cuda.empty_cache()

    if args.lock_gpu_clock:
        os.system('echo {} | sudo -S nvidia-smi -i {} -rgc'.format(args.sudo_pwd, args.cuda_device))


if __name__ == '__main__':
    try:
        mp.set_start_method("spawn")
    except:
        pass
    main()
    
    

