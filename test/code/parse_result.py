import os
import csv
import argparse

import warnings
warnings.simplefilter(action='ignore')

import pandas as pd

def get_energy(df, n_gpus):
    energy = []

    df['time_diff'] = df['timestamp'].diff()
    for i in range(n_gpus):
        df['energy_delta_{}'.format(i)] = df['time_diff'] * df['power_{}'.format(i)]
    
    for i in range(n_gpus):
        energy.append(df['energy_delta_{}'.format(i)].sum(axis=0, skipna=True))

    time = df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]
    
    freq = []
    temp = []
    for i in range(n_gpus):
        freq.append(df['sm_clock_{}'.format(i)].mean(axis=0, skipna=True) * 1000.)
        temp.append(df['temp_{}'.format(i)].mean(axis=0, skipna=True) * 1000.)
    
    return energy, time, freq, temp

def get_ncu(filename, save_to=None):
    # Count the number of rows that need to be removed
    skiprows=0
    with open(filename, 'r') as f:
        for line in f:
            vals = line.split(',"')
            if len(vals) == 1:
                skiprows += 1
            else:
                break
    
    df = pd.read_csv(filename, skiprows=skiprows)

    unique_ids = df['ID'].unique().tolist()

    summary = []
    for kernel_id in unique_ids:
        temp = {'kernel_id': kernel_id}
        df_subset = df.loc[df['ID'] == kernel_id]
        temp['kernel_name'] = df_subset['Kernel Name'].iloc[0]
        temp['block_size'] = df_subset['Block Size'].iloc[0]
        temp['grid_size'] = df_subset['Grid Size'].iloc[0]

        t_dict = pd.Series(df_subset['Metric Value'].values, index=df_subset['Metric Name']).to_dict()
        temp.update(t_dict)
        try:
            temp['max_concurrent_block'] = min(temp['Block Limit SM'], temp['Block Limit Registers'], \
                                               temp['Block Limit Shared Mem'], temp['Block Limit Warps'])
        except:
            pass
                
        summary.append(temp)
    
    if save_to is not None:
        to_write = pd.DataFrame(summary)
        to_write.to_csv(save_to, index=False)

    return summary

def main():
    # argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--path_to_folder', required=True, help='path to the folder that has all csv files to be parsed')
    parser.add_argument('--save_to', required=True, help='filename you want to save the summary result')

    args = parser.parse_args()

    if not os.path.exists(args.save_to):
        os.mkdir(args.save_to)
    
    nvml_save_to = os.path.join(args.save_to, 'nvml')
    ncu_save_to = os.path.join(args.save_to, 'ncu')
    metrics_save_to = os.path.join(args.save_to, 'metrics')

    if not os.path.exists(nvml_save_to):
        os.mkdir(nvml_save_to)
    if not os.path.exists(ncu_save_to):
        os.mkdir(ncu_save_to)
    if not os.path.exists(metrics_save_to):
        os.mkdir(metrics_save_to)

    nvml_summary = []

    for subdir, _, files in os.walk(args.path_to_folder):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith('.csv'):
                # print("Found a csv file %s", filepath)
                filename = str(file).replace(".csv", "")
                fields = filename.split("_")

                if fields[0] == 'ncu':
                    _ = get_ncu(filepath, save_to=os.path.join(ncu_save_to, str(file)))
                elif fields[0] == 'metrics':
                    _ = get_ncu(filepath, save_to=os.path.join(metrics_save_to, str(file)))
                else:
                    # NVML
                    iter_str = [x for x in fields if 'iter' in x][0]
                    iter = eval(iter_str[4:])
                    energy, time, freq, temperature = get_energy(pd.read_csv(filepath), n_gpus=1)
                    temp = {'workload': str(file), \
                            'energy': energy[0] / iter, \
                            'time': time / iter * 1000., \
                            'avg_freq': freq[0], \
                            'temp': temperature[0]}
                    nvml_summary.append(temp)
    
    # save nvml summary
    df = pd.DataFrame(nvml_summary)
    df.to_csv(os.path.join(nvml_save_to, 'nvml_parsed.csv'), index=False)

if __name__ == '__main__':
    main()
