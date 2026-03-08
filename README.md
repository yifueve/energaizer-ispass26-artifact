# EnergAIzer ISPASS'26 Artifact

Artifact of "EnergAIzer: Fast and Accurate GPU Power Estimation Framework for AI Workloads" (ISPASS'26). 

## Setup

First, clone this repository with `git clone`.

### Step 1: Download pre-collected/pre-trained files
Next, for a quick start, download the pre-collected database and measurement results following this command:
```bash
bash misc/download_files.sh
```

Alternatively, if you prefer to download them manually, please follow these steps:
1. To download the pre-collected database, navigate to `database/data`. Download [this file](https://drive.google.com/file/d/1krvqRFDnaqrJUT06V2psIua0wQr6ETAE/view?usp=sharing) and unzip the file with `tar -xzf precollected_database.tar.gz`. 
2. (Artifact Evaulation Purpose) To download pre-trained weights for the prior work comparison, navigate to `experiments_endtoend/neusight_pretrained/pretrained_torch`. Download [this file](https://drive.google.com/file/d/1XfPy67a69dZfraB3cTYY_Ac_uIGbK-a_/view?usp=drive_link) and unzip the file with `tar -xzf neusight_pretrained.tar.gz`. 

**To reproduce the results only, please follow the Step 2 below.**
If you want to use this repository for your own research and collect database from your own GPUs, please follow both the Steps 2 and 3. 

### Step 2. Quick minimal setup for artifact evaluation and reproducing the results

#### Requirements

This option does *not* require GPUs. 
Please make sure your laptop/machine can run Python 3 and Anaconda virtual environments. 

#### Run the script

1. If Anaconda is not installed, run:
```bash
cd misc
chmod +x install_conda.sh
./install_conda.sh
cd ..
```

2. Set up a virtual environment:
```bash
cd misc
source <PATH_TO_CONDA>/bin/activate
conda env create -f conda_env.yml
cd ..
```
In many cases, `PATH_TO_CONDA` will be `~/miniconda3`.
If your default option automatically activates anaconda in the shell, you can skip `source` part. 

3. Activate a virtual environment:
```bash
conda activate energaizer_env
```

### (TODO, Optional) Step 3. More setup for building your own database

#### Requirements

If you want to collect your own database from your GPUs, please make sure you have CUDA 12.0+ and Nsight Compute installed, and you have a sudo access to the machine. 
Our data collection was performed in this environment:

- CUDA 12.4
- Ubuntu 22.04.4 LTS
- cpp 11.4.0
- cmake 3.14.3

Please navigate to `database/code` and follow the instructions in [`database/code/README.md`](database/code/README.md) to complete the setup. 

## Quick start: Reproduce the results

### Run Everything

1. After activating the virtual environment, run this command. It will take 1-2 hours to be completed.
```bash
bash figures/scripts/all.sh
```

2. Once the script is completed, open the Jupyter notebook [`figures/figures.ipynb`](figures/figures.ipynb). You can execute each cell to plot the graphs corresponding to Fig. 7, 9, 10(a), and 11. 

3. If you want to check each experiment instead of running everything (or if you encounted any bug), please check below for each figure/experiment. 

### Fig. 7

The experiments needed to plot Fig. 7 can be executed with the command:
```bash
bash figures/scripts/run_single.sh
```

This command will populate the folder `experiments_single/results/` with kernel-level prediction experiment results. 
We randomly split the database into 80% train and 20% test sets. 
You can change this split ratio and increase the number of random trials in `experiments_single/bash/run_a100.sh` and `experiments_single/bash/run_a10.sh`. 

### Fig. 9

The experiments needed to plot Fig. 9 can be executed with the command:
```bash
bash figures/scripts/run_endtoend.sh
```

This comamnd will run latency and power predictions for all workloads in `test/data/workloads/all/`.
For latency  predictions, we provide comparisons with the prior arts. 

### Fig. 10(a)

The experiments needed to plot Fig. 10(a) can be executed with the command:
```bash
bash figures/scripts/run_single.sh
```

Note that this is the same as the script for Fig. 7. 

### Fig. 11

The experiments needed to plot Fig. 11 can be executed with the command:
```bash
bash figures/scripts/run_dvfs.sh
```

This command will run power predictions for the workloads in `test/data/workloads/dvfs` across the operating frequency of 510 MHz - 1410 MHz. 

## How to use

Beyond the artifact evaluation, you can use this repository for your own research involving GPU power estimation for AI workloads. 
We outline different use cases below. 

### 1. Predict energy consumption of AI models

In our artifact evaluation, we tested a few popular language and vision models, such as OPT, Qwen, Vision Transformer, and ResNet. 
However, you can test other AI models following these steps:

1. Please keep in mind that the current version only supports:
    - Single-GPU inference
    - Dense workloads that can be decomposed into dense matrix multiplications, nonlinear functions, and elementwise functions
    - Two backends for the scaled dot product attention, 'eager' (naive matrix multiplications with separate softmax) and 'flash' (FlashAttention V2)

    What it does **not** support are:
    - Mixure of experts
    - FlashDecoding kernels
    - Sparse matrix multiplications

2. Please check the step-by-step guide to prepare the workload information in [test/code/README.md](test/code/README.md). 
This will tell you how to generate JSON files that we use as inputs to EnergAIzer, similar to those in [test/data/workloads](test/data/workloads). 

3. Place your workload JSON files in one folder. 
Then, you can use the script [`experiments_endtoend/run_estimators.py`](experiments_endtoend/run_estimators.py) to generate latency and energy predictions. 
For example, to predict for NVIDIA A100-40GB-PCIE GPU at 900 MHz:
```bash
cd experiments_endtoend
python run_estimators.py \
 --workload_folder [PATH_TO_YOUR_WORKLOAD_FOLDER] \
 --result_save_to [PATH_TO_WHERE_YOU_SAVE_RESULTS] \
 --result_filename [RESULT_FILENAME] \
 --gpu_config_yaml ../config/gpu/yz8.yaml \
 --lut_config_yaml exp_config/a100_lut_config.yaml \
 --lut_folder_abs_path ../database/data \
 --no_limicro --no_neusight --no_roofline \
 --target_freq 900 \
 --flash_attention_enable \
 --flash_attention_estimate_method flashattention_v2
```

4. You can check the generated result file for latency and energy predictions.

### 2. Voltage-frequency scaling

For NVIDIA A100-40GB-PCIE GPU, you can also predict latency and energy consumption at different operating frequencies. 
To generate predictions across a range of frequencies, use this command:
```bash
cd experiments_endtoend
python run_estimators.py \
 --workload_folder [PATH_TO_YOUR_WORKLOAD_FOLDER] \
 --result_save_to [PATH_TO_WHERE_YOU_SAVE_RESULTS] \
 --result_filename [RESULT_FILENAME] \
 --gpu_config_yaml ../config/gpu/yz8.yaml \
 --lut_config_yaml exp_config/a100_dvfs_lut_config.yaml \
 --lut_folder_abs_path ../database/data \
 --no_limicro --no_neusight --no_roofline \
 --dvfs_aware \
 --dvfs_supply_voltage_json ../config/dvfs/yz8/supply_voltage.json \
 --dvfs_idle_power_json ../config/dvfs/yz8/idle_power.json
 --freq_min [MINIMUM_FREQUENCY] \
 --freq_max [MAXIMUM_FREQUENCY] \
 --freq_step [STEP_SIZE_FOR_SWEEP]
```

### 3. Design space exploration

EnergAIzer supports design space exploration for different GPU architecture configurations. 
These configurations can include different numbers of SMs, TensorCore compute throughput, and/or HBM bandwidths. 
As we assume the same energy efficiency (i.e., pJ/bit and pJ/MAC) for operations, please note that there can be larger errors when your target GPU configuration is significantly different from NVIDIA A100. 

As an example, below shows how to explore the NVIDIA H100 GPU configuration, assuming 1830 MHz @ 1.0 V (1000 mV):
```bash
cd experiments_endtoend
python run_estimators.py \
 --workload_folder [PATH_TO_YOUR_WORKLOAD_FOLDER] \
 --result_save_to [PATH_TO_WHERE_YOU_SAVE_RESULTS] \
 --result_filename [RESULT_FILENAME] \
 --multiple_config \
 --multiple_gpu_configs_yaml ../config/multiple/gpu.yaml \
 --multiple_gpu_supply_voltage_yaml ../config/multiple/supply_voltage.yaml \
 --multiple_gpu_idle_power_yaml ../config/multiple/idle_power.yaml \
 --lut_config_yaml exp_config/extrapolation_lut_config.yaml \
 --lut_folder_abs_path ../database/data \
 --no_limicro --no_neusight --no_roofline \
 --extrapolation \
 --target_gpu_config_yaml ../config/gpu/h100sxm.yaml \
 --target_supply_voltage_value 1000 \
 --target_freq 1830 \
 --target_supply_voltage_json ../config/dvfs/h100/supply_voltage.json \
 --target_idle_power_json ../config/dvfs/h100/idle_power.json 
```

If you want to explore your own GPU configurations, please create a new GPU configuration YAML file (similar to those in [`config/gpu`](config/gpu/)), set their idle power consumption profile (similar to those in [`config/dvfs/yz8/idle_power.json`](config/dvfs/yz8/idle_power.json)), then replace those paths in the above command. 

<!-- ### 4. (TODO) Add your database

### 5. (TODO) Inter-GPU communication

### 6. (TODO) Add new kernel types
 -->
