set -euo pipefail

cd experiments_endtoend

echo "Running explicit Byzantine attack demo for DVFS prediction aggregation..."

python run_dvfs_byzantine_robust.py \
  --estimated_result_path ../figures/fig11_dvfs_results/dvfs_estimator_results.csv \
  --result_save_to results/dvfs_byzantine_attack_demo \
  --result_filename dvfs_byzantine_attack_mean.csv \
  --source_estimator gee \
  --output_estimator gee_attack_mean \
  --worker_grouping all \
  --aggregation mean \
  --byzantine_workers 1 \
  --attack power_high

python run_dvfs_byzantine_robust.py \
  --estimated_result_path ../figures/fig11_dvfs_results/dvfs_estimator_results.csv \
  --result_save_to results/dvfs_byzantine_attack_demo \
  --result_filename dvfs_byzantine_attack_bro.csv \
  --source_estimator gee \
  --output_estimator gee_attack_bro \
  --worker_grouping all \
  --aggregation clipping \
  --byzantine_workers 1 \
  --attack power_high \
  --clipping_tau 3.0 \
  --inner_iterations 5

python - <<'PY'
import pandas as pd

src = pd.read_csv("../figures/fig11_dvfs_results/dvfs_estimator_results.csv")
mean = pd.read_csv("results/dvfs_byzantine_attack_demo/dvfs_byzantine_attack_mean.csv")
bro = pd.read_csv("results/dvfs_byzantine_attack_demo/dvfs_byzantine_attack_bro.csv")

combined = pd.concat(
    [src[src["estimator"] == "gee"], mean, bro],
    ignore_index=True,
)
combined.to_csv(
    "results/dvfs_byzantine_attack_demo/dvfs_byzantine_attack_demo.csv",
    index=False,
)
PY

python report_dvfs.py \
  --estimated_result_path results/dvfs_byzantine_attack_demo/dvfs_byzantine_attack_demo.csv \
  --measurement_path ../test/data/measurements/dvfs \
  --report_save_to reports/dvfs_byzantine_attack_demo \
  --freq_min 510 \
  --freq_max 1410 \
  --freq_step 45

cp results/dvfs_byzantine_attack_demo/dvfs_byzantine_attack_demo.csv ../figures/fig11_dvfs_results/dvfs_byzantine_attack_demo_estimator_results.csv
cp reports/dvfs_byzantine_attack_demo/dvfs_byzantine_attack_demo.csv ../figures/fig11_dvfs_results/dvfs_byzantine_attack_demo_report.csv

echo "Complete!"

cd ..
