import os
import sys
import argparse
import pandas as pd
import yaml

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--sudo_pwd', default='', type=str)
    parser.add_argument('--python_bin_path', default='~/miniconda3/envs/pytorch/bin/python3', type=str)

    parser.add_argument('--cuda_device', default=0, type=int)
    parser.add_argument('--gpu_clock_freq_sweep', default=False, action='store_true')
    parser.add_argument('--gpu_min_freq', default=510, type=int)
    parser.add_argument('--gpu_max_freq', default=1410, type=int)
    parser.add_argument('--gpu_freq_step', default=45, type=int)

    parser.add_argument('--run_nvml', default=False, action='store_true')
    parser.add_argument('--nvml_save_to', default='tmp/nvml/', type=str)
    parser.add_argument('--nvml_update_period', type=float, default=0.01, help='nvml update interval in s')
    parser.add_argument('--nvml_poll_clock', default=False, action='store_true')

    parser.add_argument('--run_ncu', default=False, action='store_true')
    parser.add_argument('--ncu_save_to', default='tmp/ncu/', type=str)
    parser.add_argument('--ncu_bin_path', default='/usr/local/cuda/bin/ncu', type=str)

    parser.add_argument('--run_trace', default=False, action='store_true')
    parser.add_argument('--trace_save_to', default='tmp/trace/', type=str)
    
    parser.add_argument('--model_type', type=str, choices=['Module', 'LanguageModel', 'VisionModel'], required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--config_folder', type=str, required=False)
    parser.add_argument('--precision', type=str, nargs='*', choices=['fp16', 'fp32', 'bf16'])
    parser.add_argument('--batch', type=int, nargs='*')
    parser.add_argument('--seqlen', type=int, nargs='*')
    parser.add_argument('--mode', nargs='*', choices=['prefill', 'decode'])
    parser.add_argument('--attn_backend', nargs='*', choices=['eager', 'sdpa'])

    parser.add_argument('--compile', default=False, action='store_true')

    args = parser.parse_args()

    # make folders for nvml/ncu
    if args.run_nvml and (not os.path.exists(args.nvml_save_to)):
        os.mkdir(args.nvml_save_to)
    if args.run_ncu and (not os.path.exists(args.ncu_save_to)):
        os.mkdir(args.ncu_save_to)
    if args.run_trace and (not os.path.exists(args.trace_save_to)):
        os.mkdir(args.trace_save_to)

    # enumerate the iteration space
    if args.config_folder is not None:
        all_configs = [os.path.join(args.config_folder, f) for f in os.listdir(args.config_folder) if os.path.isfile(os.path.join(args.config_folder, f))]
    else:
        all_configs = ['']
    all_precs = args.precision
    all_batches = args.batch
    all_seqlen = args.seqlen
    all_mode = args.mode
    all_attn_backend = args.attn_backend

    if args.gpu_clock_freq_sweep:
        all_freqs = list(range(args.gpu_min_freq, args.gpu_max_freq+1, args.gpu_freq_step))
    else:
        all_freqs = [-1]

    # calls
    for c in all_configs:
        for p in all_precs:
            for b in all_batches:
                for s in all_seqlen:
                    for m in all_mode:
                        for a in all_attn_backend:
                            for f in all_freqs:

                                # command
                                if args.model == 'GPTOSSModel':
                                    cmd = '{} run_gptoss.py'.format(args.python_bin_path)
                                else:
                                    cmd = '{} run_model.py'.format(args.python_bin_path)

                                if len(args.sudo_pwd) > 0:
                                    cmd += ' --sudo_pwd {}'.format(args.sudo_pwd)
                                
                                cmd += ' --cuda_device {}'.format(args.cuda_device)
                                if f > 0:
                                    cmd += ' --lock_gpu_clock --gpu_clock_freq {}'.format(f)
                                
                                cmd += ' --model_type {} --model {} --precision {} --batch {} --seqlen {} --mode {} --attn_backend {}'.format(args.model_type, args.model, \
                                                                                                                                              p, b, s, m, a)

                                if len(c) > 0:
                                    cmd += ' --config_file {}'.format(c)
                                
                                if args.compile:
                                    cmd += ' --compile'

                                config_name = c.split('/')[-1][:-5]
                                filename = '{}_{}_p{}_b{}'.format(args.model.lower(), config_name, p, b)
                                if (args.model_type == 'Module') or (args.model_type == 'LanguageModel'):
                                    filename += '_s{}_mode{}_attn{}'.format(s, m, a)
                                if f > 0:
                                    filename += '_freq{}'.format(f)
                                filename += '.csv'

                                if args.run_nvml:
                                    nvml_cmd = 'CUDA_VISIBLE_DEVICES={} '.format(args.cuda_device)
                                    nvml_cmd += cmd
                                    nvml_cmd += ' --run_nvml --nvml_save_to {} --nvml_update_period {}'.format(os.path.join(args.nvml_save_to, filename), args.nvml_update_period)
                                    if args.nvml_poll_clock:
                                        nvml_cmd += ' --nvml_poll_clock'
                                    
                                    # print(nvml_cmd)
                                    os.system(nvml_cmd)

                                if args.run_ncu:
                                    ncu_cmd = 'echo {} | sudo -S CUDA_VISIBLE_DEVICES={} '.format(args.sudo_pwd, args.cuda_device)
                                    ncu_filename = 'ncu_' + filename
                                    ncu_cmd += '/usr/local/cuda/bin/ncu --log-file {} --csv --metrics sm__cycles_elapsed '.format(os.path.join(args.ncu_save_to, ncu_filename)) 
                                    ncu_cmd += cmd
                                    ncu_cmd += ' --ncu'

                                    # print(ncu_cmd)
                                    os.system(ncu_cmd)

                                if args.run_trace:
                                    trace_cmd = 'CUDA_VISIBLE_DEVICES={} '.format(args.cuda_device)
                                    trace_cmd += cmd
                                    trace_cmd += ' --trace --trace_save_to {}'.format(os.path.join(args.trace_save_to, filename))

                                    # print(trace_cmd)
                                    os.system(trace_cmd)

if __name__ == '__main__':
    main()
