import os
import sys
import argparse
import copy

import warnings
warnings.filterwarnings('ignore')

import datetime

import pandas as pd
import numpy as np
import yaml

from sklearn.utils import shuffle

from gee_estimator import GeeEstimator
from limicro_estimator import LiEstimator
from neusight_estimator import NeuSightEstimator
from throughput_estimator import ThroughputEstimator

def create_dummy_lut(train_data_path, save_to, workload_type):
    # Create a lut config
    lut_config = {}

    if workload_type == 'gemm':
        lut_config['gemm'] = {}
        
        # The config should reflect the data
        tmp_df = pd.read_csv(train_data_path)
        precM = tmp_df['precM'].iloc[0]
        precA = tmp_df['precA'].iloc[0]
        useTensorCore = tmp_df['useTensorCore'].iloc[0]
        if useTensorCore:
            lut_config['gemm']['tc'] = {}
            lut_config['gemm']['tc']['{}_{}'.format(precM, precA)] = [{'path': 'df_train.csv', 'prepared': False, 'main': True, 'use_for_model': True, 'use_for_kernel': True, 'require_annotation': False}]
        else:
            lut_config['gemm']['cuda'] = {}
            lut_config['gemm']['cuda']['{}_{}'.format(precM, precA)] = [{'path': 'df_train.csv', 'prepared': False, 'main': True, 'use_for_model': True, 'use_for_kernel': True, 'require_annotation': False}]

    elif workload_type == 'conv2d':
        lut_config['conv2d'] = {}
        
        # The config should reflect the data
        tmp_df = pd.read_csv(train_data_path)
        precM = tmp_df['precM'].iloc[0]
        precA = tmp_df['precA'].iloc[0]
        useTensorCore = tmp_df['useTensorCore'].iloc[0]
        if useTensorCore:
            lut_config['conv2d']['tc'] = {}
            lut_config['conv2d']['tc']['{}_{}'.format(precM, precA)] = [{'path': 'df_train.csv', 'prepared': False, 'main': True, 'use_for_model': True, 'use_for_kernel': True, 'require_annotation': False}]
        else:
            lut_config['conv2d']['cuda'] = {}
            lut_config['conv2d']['cuda']['{}_{}'.format(precM, precA)] = [{'path': 'df_train.csv', 'prepared': False, 'main': True, 'use_for_model': True, 'use_for_kernel': True, 'require_annotation': False}]


    elif workload_type == 'softmax':
        lut_config['softmax'] = {}
        tmp_df = pd.read_csv(train_data_path)
        prec = tmp_df['prec'].iloc[0]

        lut_config['softmax'][prec] = [{'path': 'df_train.csv', 'prepared': False, 'main': True, 'use_for_model': True, 'use_for_kernel': True, 'require_annotation': False}]
    
    elif workload_type == 'layernorm':
        lut_config['layernorm'] = {}
        tmp_df = pd.read_csv(train_data_path)
        prec = tmp_df['prec'].iloc[0]

        lut_config['layernorm'][prec] = [{'path': 'df_train.csv', 'prepared': False, 'main': True, 'use_for_model': True, 'use_for_kernel': True, 'require_annotation': False}]

    elif workload_type == 'elementwise':
        lut_config['elementwise'] = []
        tmp_df = pd.read_csv(train_data_path)
        lut_config['elementwise'].append({'path': 'df_train.csv', 'prepared': False, 'main': True, 'use_for_model': True, 'use_for_kernel': True, 'require_annotation': False})

    elif workload_type == 'flashattention':
        lut_config['flashattention_v2'] = {}
        lut_config['flashattention_v2']['bf16'] = []
        tmp_df = pd.read_csv(train_data_path)
        lut_config['flashattention_v2']['bf16'].append({'path': 'df_train.csv', 'prepared': False, 'main': True, 'use_for_model': True, 'use_for_kernel': True, 'require_annotation': False})

    with open(os.path.join(save_to, 'lut.yaml'), 'w') as f:
        yaml.dump({'lut_config': lut_config}, f)

    del tmp_df

def main():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--path_to_data', type=str, required=True, help='Path to the .csv file of data')
    parser.add_argument('--workload_type', default='gemm', choices=['gemm', 'softmax', 'layernorm', 'conv2d', 'elementwise', 'flashattention'], help='Workload type to test (GEMM, Conv2D, Softmax, LayerNorm, Misc Elementwise)')
    parser.add_argument('--data_subset_option', default='all', choices=['all', 'small', 'smallM', 'smallN', 'smallK', 'batch1', 'regular', 'large', 'notlarge'], help='Only test with a subset of data for GEMM workloads')
    parser.add_argument('--transpose_options', nargs='*', help='Only use specific transpose conditions stated for GEMM workloads. If left empty, all transpose options in the dataset will be used.')
    parser.add_argument('--test_fraction', type=float, required=True, help='Fraction of data to be used for testing')
    parser.add_argument('--save_to', type=str, required=True, help='Path to the folder to save the results')
    parser.add_argument('--random_seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--remake_dataset_if_exists', action='store_true', default=False, \
                        help='override existing dataset and recreate one if already exists in the path')
    parser.add_argument('--delete_data_after_done', action='store_true', default=False, help='Delete train/test data csv files after finishing evaluation')

    # GEE Related
    parser.add_argument('--use_gee', action='store_true', default=False, help='Run Gpu-Energy-Estimator (GEE)')
    parser.add_argument('--gee_gpu_config', type=str, help='GPU configuration file to be used for GEE', required=False)
    parser.add_argument('--gee_lookup_method', type=str, choices=['v3'], nargs='*', required=False, \
                        help='Look-up method (NCU or No NCU) for Gee inference')
    parser.add_argument('--gee_use_entire_ref', nargs='*', type=int, choices=[0, 1], help='Using the entire reference table (no trimming based on K/blocks condition) for GEE V2.3')
    parser.add_argument('--gee_dvfs_aware', nargs='*', type=int, choices=[0, 1],  help='Enable DVFS aware power model')
    parser.add_argument('--gee_dvfs_idle_power_json', type=str, help='Path to the .json file for idle power information')
    parser.add_argument('--gee_dvfs_supply_voltage_json', type=str, help='Path to the .json file for supply voltage information')

    # Li-MICRO23 Related
    parser.add_argument('--use_limicro', action='store_true', default=False, help='Run Li-MICRO23 Estimator')
    parser.add_argument('--limicro_estimation_target', type=str, choices=['throughput', 'endtoend'], default='endtoend')

    # NeuSight Related
    parser.add_argument('--use_neusight', action='store_true', default=False, help='Run NeuSight (ASPLOS25) Estimator')
    parser.add_argument('--neusight_gpu_config', type=str, help='GPU configuration file to be used for NeuSight', required=False)
    parser.add_argument('--neusight_train_config', type=str, help='NeuSight model/train configuration file', required=False)
    parser.add_argument('--neusight_max_iterations', type=int, default=1, help='Allow retraining of NeuSight performance model until the desired error level is reached')
    parser.add_argument('--neusight_perf_target_mape', type=float, default=25.0, help='Target MAPE (mean average percent error) for NeuSight Performance Model')

    # Throughput/Analytical Related
    parser.add_argument('--use_analytical', action='store_true', default=False, help='Use analytical/thorughput (i.e., roofline) model')
    parser.add_argument('--analytical_mode', type=str, choices=['naive_roofline', 'loopnest_roofline', 'timeline_analytical'], nargs='*', \
                        help='Analytical (no data) model mode')
    parser.add_argument('--analytical_gpu_config', type=str, help='GPU configuration file to be used')

    args = parser.parse_args()

    """ 
    Folder Structure
    path_to_data
    |- lut
    |- |- df_test.csv (test dataset)
    |- |- df_train.csv (train_dataset)
    |- gee (if GEE is used)
    |- |- estimation_result.csv (report generated)
    |- |- lut.yaml (LUT configuration file for initializing Gee)
    |- limicro (if LiMicro is used)
    |- |- estimation_result.csv
    |- neusight (if NeuSight is used)
    |- |- estimation_result.csv
    """

    print("Result will be save to {}".format(args.save_to))

    if not os.path.exists(args.save_to):
        try:
            os.mkdir(args.save_to)
        except:
            print("Error while creating folder {} - check the path and parent directories".format(args.save_to))
            exit()

    if not os.path.exists(os.path.join(args.save_to, 'lut')):
        os.mkdir(os.path.join(args.save_to, 'lut'))

    #### Random Seed ####
    if args.random_seed is not None:
        np.random.seed(seed=args.random_seed)
        print("Using random seed {}".format(args.random_seed))
        seed = args.random_seed
    else:
        today = datetime.date.today()
        seed = int(today.strftime("%Y%m%d"))
        np.random.seed(seed=seed)
        print("Using random seed {}".format(seed))

    #### Data Split ####
    
    # Check if previously train/test data already created
    train_data_path = os.path.join(args.save_to, 'lut', 'df_train.csv')
    test_data_path = os.path.join(args.save_to, 'lut', 'df_test.csv')
    data_exists = os.path.exists(train_data_path) and os.path.exists(test_data_path)
    
    if (not data_exists) or (args.remake_dataset_if_exists):
        print("Generating train/test splitted dataset..")
        df = pd.read_csv(args.path_to_data)

        if (args.workload_type == 'gemm') or (args.workload_type == 'conv2d'):
            df.drop_duplicates(subset=['batch','dimM','dimN','dimK','precM','precA','useTensorCore'], inplace=True)

            # Check if regular shape only is set
            # if args.regular_problem_shapes_only:
            #     df = df.loc[(df['dimM'] >= 128) & (df['dimN'] >= 128) & (df['dimK'] >=128)].reset_index(drop=True)

            # if args.batched_matmul_only:
            #     df = df.loc[(df['batch'] > 1)].reset_index(drop=True)

            if args.data_subset_option == 'regular':
                df = df.loc[(df['dimM'] >= 128) & (df['dimN'] >= 128) & (df['dimK'] >=128)].reset_index(drop=True)
            elif args.data_subset_option == 'small':
                df = df.loc[(df['dimM'] < 128) | (df['dimN'] < 128) | (df['dimK'] < 128)].reset_index(drop=True)
            elif args.data_subset_option == 'smallM':
                df = df.loc[(df['dimM'] < 128)].reset_index(drop=True)
            elif args.data_subset_option == 'smallN':
                df = df.loc[(df['dimN'] < 128)].reset_index(drop=True)
            elif args.data_subset_option == 'smallK':
                df = df.loc[(df['dimK'] < 128)].reset_index(drop=True)
            elif args.data_subset_option == 'batch1':
                df = df.loc[(df['batch'] == 1)].reset_index(drop=True)
            elif args.data_subset_option == 'large':
                df = df.loc[(df['dimM'] >= 512) & (df['dimN'] >= 512) & (df['dimK'] >= 512)].reset_index(drop=True)
            elif args.data_subset_option == 'notlarge':
                df = df.loc[(df['dimM'] < 2**17) & (df['dimN'] < 2**17) & (df['dimK'] < 2**17)].reset_index(drop=True)

            # Transpose options
            if (args.transpose_options is not None):
                df = df.loc[(df['trans'].isin(args.transpose_options))].reset_index(drop=True)
        elif (args.workload_type == 'softmax') or (args.workload_type == 'layernorm'):
            df.drop_duplicates(subset=['batch', 'dim', 'prec'], inplace=True)
        elif (args.workload_type == 'conv2d'):
            df.drop_duplicates(subset=['b', 'm', 'c', 'hw', 'rs', 'stride', 'padding'], inplace=True)
        elif (args.workload_type == 'elementwise'):
            df.drop_duplicates(subset=['op', 'prec', 'dim'], inplace=True)
        elif(args.workload_type == 'flashattention'):
            df.drop_duplicates(subset=['batch', 'n_head', 'seq_len', 'head_dim', 'prec'], inplace=True)

        df_shuffled = shuffle(df)
        df_shuffled.reset_index(inplace=True, drop=True)

        n_rows = len(df_shuffled)
        split = int(n_rows * args.test_fraction)

        df_test = df_shuffled.iloc[0:split]
        df_train = df_shuffled.iloc[split:n_rows]
        df_train.to_csv(train_data_path, index=False)
        df_test.to_csv(test_data_path, index=False)
            

    #### Estimators ####
    if args.use_gee:
        print("Calling GEE Estimator..")
        gee_iteration_space = []
        
        for ref_method in args.gee_use_entire_ref:
            for dvfs_method in args.gee_dvfs_aware:
                config = ((ref_method==1), (dvfs_method==1))
                gee_iteration_space.append(config)

        gee_iteration_space = list(set(gee_iteration_space))

        for config in gee_iteration_space:
            ref_method = config[0]
            dvfs_method = config[1]

            print("--GEE Use Entire Ref Option: {}".format(ref_method))
            print("--GEE DVFS Aware Option: {}".format(dvfs_method))

            subfoldername = 'gee_{}_ref{}_dvfs{}'.format('v3', ref_method, dvfs_method)
            if not os.path.exists(os.path.join(args.save_to, subfoldername)):
                os.mkdir(os.path.join(args.save_to, subfoldername))

            create_dummy_lut(train_data_path, os.path.join(args.save_to, subfoldername), args.workload_type)

            estimator_instance = GeeEstimator(train_data_path, test_data_path, gpu_config=args.gee_gpu_config, \
                                              lut_config=os.path.join(args.save_to, subfoldername, 'lut.yaml'), \
                                              lut_parent_path=args.save_to,
                                              use_entire_ref=ref_method, \
                                              dvfs_aware=dvfs_method, dvfs_idle_power_json=args.gee_dvfs_idle_power_json, \
                                              dvfs_supply_voltage_json=args.gee_dvfs_supply_voltage_json, \
                                              workload_type=args.workload_type)
            
            estimator_instance.generate_test_report(['energy', 'time'], os.path.join(args.save_to, subfoldername, 'estimation_result.csv'), True)

    if args.use_limicro:
        print("Calling Li (MICRO23) Estimator..")
        if not os.path.exists(os.path.join(args.save_to, 'limicro')):
            os.mkdir(os.path.join(args.save_to, 'limicro'))

        estimator_instance = LiEstimator(train_data_path, test_data_path, workload_type=args.workload_type, estimation_target=args.limicro_estimation_target)
        estimator_instance.generate_test_report(['energy', 'time'], os.path.join(args.save_to, 'limicro', 'estimation_result.csv'), True)

    if args.use_neusight:
        print("Calling NeuSight (ASPLOS25) Estimator..")
        if not os.path.exists(os.path.join(args.save_to, 'neusight')):
            os.mkdir(os.path.join(args.save_to, 'neusight'))

        create_dummy_lut(train_data_path, os.path.join(args.save_to, 'neusight'), args.workload_type)

        best_estimator = None
        best_mape = 1000

        for i in range(args.neusight_max_iterations):
            _estimator = NeuSightEstimator(train_data_path, test_data_path, gpu_config=args.neusight_gpu_config, \
                                           lut_config=os.path.join(args.save_to, 'neusight', 'lut.yaml'), \
                                           lut_parent_path=args.save_to, \
                                           workload_type=args.workload_type)
            _estimator.set_train_config_from_yaml(args.neusight_train_config)

            print("Iteration {}: Train NeuSight Performance Estimation Model".format(i+1))
            _estimator.train('time', False, True)
            _estimator.test('time')
            _estimator.test_df['percent_error_time'] = np.abs(_estimator.test_df['time'] - _estimator.test_df['time_estimate']) / _estimator.test_df['time'] * 100.
            mape_time = _estimator.test_df['percent_error_time'].mean()
            print(" --> Performance MAPE: {:.4f} % (Target MAPE: {:.4f} %)".format(mape_time, args.neusight_perf_target_mape))

            if mape_time < best_mape:
                best_estimator = copy.deepcopy(_estimator)
                best_mape = mape_time
                print(" --> Best MAPE so far..")

            if mape_time < args.neusight_perf_target_mape:
                del _estimator
                break

            if i == (args.neusight_max_iterations - 1):
                del _estimator

        estimator_instance = best_estimator

        print("Training NeuSight Energy Estimation Model")
        estimator_instance.train('energy', False, False)

        estimator_instance.generate_test_report(['energy', 'time'], os.path.join(args.save_to, 'neusight', 'estimation_result.csv'), True)

    if args.use_analytical:
        print("Calling Analytical/Throughput Estimator..")
        for method in args.analytical_mode:
            print("--Analytical method: {}".format(method))

            subfoldername = 'analytical_{}'.format(method)
            if not os.path.exists(os.path.join(args.save_to, subfoldername)):
                os.mkdir(os.path.join(args.save_to, subfoldername))
            
            create_dummy_lut(train_data_path, os.path.join(args.save_to, subfoldername), args.workload_type)

            estimator_instance = ThroughputEstimator(train_data_path, test_data_path, gpu_config=args.analytical_gpu_config, \
                                                     throughput_mode=method, lut_config=os.path.join(args.save_to, subfoldername, 'lut.yaml'), \
                                                     lut_parent_path=args.save_to, workload_type=args.workload_type)
            estimator_instance.generate_test_report(['energy', 'time'], os.path.join(args.save_to, subfoldername, 'estimation_result.csv'), True)

    print("Finished Evaluation. Exit.")

    if args.delete_data_after_done:
        os.remove(train_data_path)
        os.remove(test_data_path)
    
if __name__ == '__main__':
    main()
