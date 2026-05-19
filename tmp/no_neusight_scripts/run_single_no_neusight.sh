cd experiments_single

echo "Running A100 kernel-level predictions without NeuSight..."
bash ../tmp/no_neusight_scripts/experiments_single_bash/run_a100_no_neusight.sh
echo "Complete!\n"

echo "Running A10 kernel-level predictions without NeuSight..."
bash ../tmp/no_neusight_scripts/experiments_single_bash/run_a10_no_neusight.sh
echo "Complete!\n"

cd ..
