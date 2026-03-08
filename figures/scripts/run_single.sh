cd experiments_single

echo "Running A100 kernel-level predictions..."
bash bash/run_a100.sh
echo "Complete!\n"

echo "Running A10 kernel-level predictions..."
bash bash/run_a10.sh
echo "Complete!\n"

cd ..
