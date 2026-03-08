cd experiments_endtoend

echo "Running A100 end-to-end predictions..."
python run_estimators.py --workload_folder ../test/data/workloads/all --result_save_to results/a100_endtoend --result_filename all.csv --gpu_config_yaml ../config/gpu/yz8.yaml --lut_config_yaml exp_config/a100_lut_config.yaml --lut_folder_abs_path ../database/data --no_roofline --target_freq 900 --neusight_gemm_train_config neusight_pretrained/config/neusight_config_gemm.yaml --neusight_nonlinear_train_config neusight_pretrained/config/neusight_config_nonlinear.yaml --neusight_pretrained --neusight_pretrained_model_path neusight_pretrained/pretrained_torch/a100/pretrained.yaml
python report.py --estimated_result_path results/a100_endtoend/all.csv --measurement_path ../test/data/measurements/a100 --report_save_to reports/a100_endtoend
echo "Complete!"

echo "Running A10 end-to-end predictions..."
python run_estimators.py --workload_folder ../test/data/workloads/all --result_save_to results/a10_endtoend --result_filename all.csv --gpu_config_yaml ../config/gpu/a10.yaml --lut_config_yaml exp_config/a10_lut_config.yaml --lut_folder_abs_path ../database/data --no_roofline --target_freq 900 --neusight_gemm_train_config neusight_pretrained/config/neusight_config_gemm.yaml --neusight_nonlinear_train_config neusight_pretrained/config/neusight_config_nonlinear.yaml --neusight_pretrained --neusight_pretrained_model_path neusight_pretrained/pretrained_torch/a10/pretrained.yaml
python report.py --estimated_result_path results/a10_endtoend/all.csv --measurement_path ../test/data/measurements/a10 --report_save_to reports/a10_endtoend
echo "Complete!"

cd ..