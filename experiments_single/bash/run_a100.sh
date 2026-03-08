#! /bin/bash
trials=1
# timestamp=$(date +%Y%m%d_%H%M%S)
timestamp="artifact"

# FP32 Single-precision CUDA Core in yz8
folder="results/yz8_fp32_cuda_${timestamp}"
mkdir -p $folder

lut_path="../database/data/yz8_sgemm_freq900_noflush_lut_v2.csv"

for ((i=1; i<=$trials; i++))
do
    python run_empirical_models.py --path_to_data ${lut_path} --workload_type gemm --test_fraction 0.2 --save_to "${folder}/seed${i}" --data_subset_option all --remake_dataset_if_exists --random_seed ${i} --use_gee --gee_gpu_config ../config/gpu/yz8.yaml --gee_lookup_method v3 --gee_use_entire_ref 0 --gee_dvfs_aware 0 --gee_dvfs_idle_power_json ../config/dvfs/yz8/idle_power.json --gee_dvfs_supply_voltage_json ../config/dvfs/yz8/supply_voltage.json --use_limicro --use_neusight --neusight_gpu_config ../config/gpu/yz8.yaml --neusight_train_config neusight_config/neusight_config_gemm.yaml
done

# BF16 Half-precision Tensor Core in yz8
folder="results/yz8_bf16bf16_tc_${timestamp}"
mkdir -p $folder

lut_path="../database/data/yz8_gemmex_bf16bf16_freq900_noflush_lut_v2.csv"

for ((i=1; i<=$trials; i++))
do
    python run_empirical_models.py --path_to_data ${lut_path} --workload_type gemm --test_fraction 0.2 --save_to "${folder}/seed${i}" --data_subset_option all --remake_dataset_if_exists --random_seed ${i} --use_gee --gee_gpu_config ../config/gpu/yz8.yaml --gee_lookup_method v3 --gee_use_entire_ref 0 --gee_dvfs_aware 0 --gee_dvfs_idle_power_json ../config/dvfs/yz8/idle_power.json --gee_dvfs_supply_voltage_json ../config/dvfs/yz8/supply_voltage.json --use_limicro --use_neusight --neusight_gpu_config ../config/gpu/yz8.yaml --neusight_train_config neusight_config/neusight_config_gemm.yaml
done

# BF16 Softmax
folder="results/yz8_bf16_softmax_${timestamp}"
mkdir -p $folder

lut_path="../database/data/yz8_softmax_bf16_freq900_lut.csv"

for ((i=1; i<=$trials; i++))
do
    python run_empirical_models.py --path_to_data ${lut_path} --workload_type softmax --test_fraction 0.2 --save_to "${folder}/seed${i}" --data_subset_option all --remake_dataset_if_exists --random_seed ${i} --use_gee --gee_gpu_config ../config/gpu/yz8.yaml --gee_lookup_method v3 --gee_use_entire_ref 0 --gee_dvfs_aware 0 --use_limicro --use_neusight --neusight_gpu_config ../config/gpu/yz8.yaml --neusight_train_config neusight_config/neusight_config_nonlinear.yaml
done

# FP32 Softmax
folder="results/yz8_fp32_softmax_${timestamp}"
mkdir -p $folder

lut_path="../database/data/yz8_softmax_fp32_freq900_lut.csv"

for ((i=1; i<=$trials; i++))
do
    python run_empirical_models.py --path_to_data ${lut_path} --workload_type softmax --test_fraction 0.2 --save_to "${folder}/seed${i}" --data_subset_option all --remake_dataset_if_exists --random_seed ${i} --use_gee --gee_gpu_config ../config/gpu/yz8.yaml --gee_lookup_method v3 --gee_use_entire_ref 0 --gee_dvfs_aware 0 --use_limicro --use_neusight --neusight_gpu_config ../config/gpu/yz8.yaml --neusight_train_config neusight_config/neusight_config_nonlinear.yaml
done

# BF16 LayerNorm
folder="results/yz8_bf16_layernorm_${timestamp}"
mkdir -p $folder

lut_path="../database/data/yz8_layernorm_bf16_freq900_lut.csv"

for ((i=1; i<=$trials; i++))
do
    python run_empirical_models.py --path_to_data ${lut_path} --workload_type layernorm --test_fraction 0.2 --save_to "${folder}/seed${i}" --data_subset_option all --remake_dataset_if_exists --random_seed ${i} --use_gee --gee_gpu_config ../config/gpu/yz8.yaml --gee_lookup_method v3 --gee_use_entire_ref 0 --gee_dvfs_aware 0 --use_limicro --use_neusight --neusight_gpu_config ../config/gpu/yz8.yaml --neusight_train_config neusight_config/neusight_config_nonlinear.yaml
done

# FP32 LayerNorm
folder="results/yz8_fp32_layernorm_${timestamp}"
mkdir -p $folder

lut_path="../database/data/yz8_layernorm_fp32_freq900_lut.csv"

for ((i=1; i<=$trials; i++))
do
    python run_empirical_models.py --path_to_data ${lut_path} --workload_type layernorm --test_fraction 0.2 --save_to "${folder}/seed${i}" --data_subset_option all --remake_dataset_if_exists --random_seed ${i} --use_gee --gee_gpu_config ../config/gpu/yz8.yaml --gee_lookup_method v3 --gee_use_entire_ref 0 --gee_dvfs_aware 0 --use_limicro --use_neusight --neusight_gpu_config ../config/gpu/yz8.yaml --neusight_train_config neusight_config/neusight_config_nonlinear.yaml
done

# BF16 Conv2d
folder="results/yz8_conv2d_bf16bf16_tc_${timestamp}"
mkdir -p $folder

lut_path="../database/data/yz8_conv2d_bf16_freq900_lut.csv"

for ((i=1; i<=$trials; i++))
do
    python run_empirical_models.py --path_to_data ${lut_path} --workload_type conv2d --test_fraction 0.2 --save_to "${folder}/seed${i}" --remake_dataset_if_exists --random_seed ${i} --use_gee --gee_gpu_config ../config/gpu/yz8.yaml --gee_lookup_method v3 --gee_use_entire_ref 0 --gee_dvfs_aware 0 --gee_dvfs_idle_power_json ../config/dvfs/yz8/idle_power.json --gee_dvfs_supply_voltage_json ../config/dvfs/yz8/supply_voltage.json --use_limicro --use_neusight --neusight_gpu_config ../config/gpu/yz8.yaml --neusight_train_config neusight_config/neusight_config_gemm.yaml
done

# Elementwise
folder="results/yz8_elementwise_${timestamp}"
mkdir -p $folder

lut_path="../database/data/yz8_misc_elementwise_freq900_lut.csv"

for ((i=1; i<=$trials; i++))
do
    python run_empirical_models.py --path_to_data ${lut_path} --workload_type elementwise --test_fraction 0.2 --save_to "${folder}/seed${i}" --remake_dataset_if_exists --random_seed ${i} --use_gee --gee_gpu_config ../config/gpu/yz8.yaml --gee_lookup_method v3 --gee_use_entire_ref 0 --gee_dvfs_aware 0 --gee_dvfs_idle_power_json ../config/dvfs/yz8/idle_power.json --gee_dvfs_supply_voltage_json ../config/dvfs/yz8/supply_voltage.json --use_limicro --use_neusight --neusight_gpu_config ../config/gpu/yz8.yaml --neusight_train_config neusight_config/neusight_config_nonlinear.yaml
done

# FlashAttention
folder="results/yz8_flashattention_${timestamp}"
mkdir -p $folder

lut_path="../database/data/yz8_flashattention_freq900_lut.csv"

for ((i=1; i<=$trials; i++))
do
    python run_empirical_models.py --path_to_data ${lut_path} --workload_type flashattention --test_fraction 0.2 --save_to "${folder}/seed${i}" --remake_dataset_if_exists --random_seed ${i} --use_gee --gee_gpu_config ../config/gpu/yz8.yaml --gee_lookup_method v3 --gee_use_entire_ref 0 --gee_dvfs_aware 0 --gee_dvfs_idle_power_json ../config/dvfs/yz8/idle_power.json --gee_dvfs_supply_voltage_json ../config/dvfs/yz8/supply_voltage.json
done

