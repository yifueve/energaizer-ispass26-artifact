set -euo pipefail

cd experiments_endtoend

echo "Running Byzantine-robust DVFS prediction aggregation..."
python run_dvfs_byzantine_robust.py \
  --estimated_result_path ../figures/fig11_dvfs_results/dvfs_estimator_results.csv \
  --result_save_to results/dvfs_byzantine_robust \
  --result_filename dvfs_byzantine_robust.csv \
  --source_estimator gee \
  --output_estimator gee_bro \
  --aggregation clipping \
  --clipping_tau 10.0 \
  --inner_iterations 5

python report_dvfs.py \
  --estimated_result_path results/dvfs_byzantine_robust/dvfs_byzantine_robust.csv \
  --measurement_path ../test/data/measurements/dvfs \
  --report_save_to reports/dvfs_byzantine_robust \
  --freq_min 510 \
  --freq_max 1410 \
  --freq_step 45

cp results/dvfs_byzantine_robust/dvfs_byzantine_robust.csv ../figures/fig11_dvfs_results/dvfs_byzantine_robust_estimator_results.csv
cp reports/dvfs_byzantine_robust/dvfs_byzantine_robust.csv ../figures/fig11_dvfs_results/dvfs_byzantine_robust_report.csv

echo "Complete!"

cd ..
