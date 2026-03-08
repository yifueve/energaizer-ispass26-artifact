cd experiments_endtoend

echo "Running DVFS experiments..."
python run_estimators.py --workload_folder ../test/data/workloads/dvfs --result_save_to results/dvfs --result_filename dvfs.csv --gpu_config_yaml ../config/gpu/yz8.yaml --lut_config_yaml exp_config/a100_dvfs_lut_config.yaml --lut_folder_abs_path ../database/data --no_roofline --no_limicro --no_neusight --dvfs_aware --dvfs_supply_voltage_json ../config/dvfs/yz8/supply_voltage.json --dvfs_idle_power_json ../config/dvfs/yz8/idle_power.json --freq_min 510 --freq_max 1410 --freq_step 45
python report_dvfs.py --estimated_result_path results/dvfs/dvfs.csv --measurement_path ../test/data/measurements/dvfs --report_save_to reports/dvfs --freq_min 510 --freq_max 1410 --freq_step 45
echo "Complete!"

cd ..