#!/bin/bash
#SBATCH --job-name=conquernet-real
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=outputs/logs/real_data_%j.out
#SBATCH --error=outputs/logs/real_data_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODULE_INIT="${MODULE_INIT:-/public1/soft/modules/module.sh}"

if [[ -f "$MODULE_INIT" ]]; then
  source "$MODULE_INIT"
  module load "${GCC_MODULE:-gcc/12.2}"
fi

mkdir -p "$PROJECT_DIR/outputs/logs" "$PROJECT_DIR/outputs/real_data"
cd "$PROJECT_DIR/python"
"$PYTHON_BIN" -u real_data_benchmark.py \
  --dataset all \
  --output ../outputs/real_data/neural_results.csv
