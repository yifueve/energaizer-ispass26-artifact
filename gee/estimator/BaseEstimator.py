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

class BaseEstimator():
    def __init__(self, op, ops_supported, gpu_config=None, \
                 dvfs_aware=False, dvfs_inference_mode='single_source', dvfs_supply_voltage={}, dvfs_idle_power={}, \
                 dvfs_max_hbm_freq=2619, dvfs_max_core_freq=1980, dvfs_max_core_voltage=1400, \
                 multiple_configs=False, gpu_configs={}, dvfs_idle_power_configs={}, dvfs_supply_voltage_configs={}):
        
        self.op = op
        self.ops_supported = ops_supported

        self.gpu_config = gpu_config
        self.dvfs_aware = dvfs_aware
        self.dvfs_inference_mode = dvfs_inference_mode
        self.dvfs_supply_voltage = dvfs_supply_voltage
        self.dvfs_idle_power = dvfs_idle_power

        # For extrapolation into GPUs with different max freq/voltage
        self.dvfs_max_hbm_freq = dvfs_max_hbm_freq
        self.dvfs_max_core_freq = dvfs_max_core_freq
        self.dvfs_max_core_voltage = dvfs_max_core_voltage

        # Needs to be instantiated in the child class
        self.kernel_parser = None
        self.analytical_model = None
        self.kernel_predictor = None
        self.model_database = None
        self.kernel_database = None

        # Database has multiple GPUs -> lut config should specify the GPU tag
        self.multiple_configs = multiple_configs
        self.gpu_configs = gpu_configs
        self.dvfs_idle_power_configs = dvfs_idle_power_configs
        self.dvfs_supply_voltage_configs = dvfs_supply_voltage_configs

    def build(self):
        raise NotImplementedError()
    
    def build_kernel_predictor(self):
        raise NotImplementedError()
    
    def predict(self):
        raise NotImplementedError()
    
    def predict_kernel(self):
        raise NotImplementedError()

    def get_references(self):
        raise NotImplementedError()
    
    
