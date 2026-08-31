#!/bin/bash
#SBATCH --job-name=tabpfn-reference
#SBATCH --array=0-2
#SBATCH --cpus-per-task=48
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/logs/tabpfn_%A_%a.out
#SBATCH --error=outputs/logs/tabpfn_%A_%a.err

set -euo pipefail

: "${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to the directory containing the pinned checkpoint}"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODULE_INIT="${MODULE_INIT:-/public1/soft/modules/module.sh}"
DATASETS=(bmi_male bmi_female california_housing)
DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"

if [[ -f "$MODULE_INIT" ]]; then
  source "$MODULE_INIT"
  module load "${GCC_MODULE:-gcc/12.2}"
fi

mkdir -p "$PROJECT_DIR/outputs/logs" "$PROJECT_DIR/outputs/real_data"
cd "$PROJECT_DIR/python"
"$PYTHON_BIN" -u tabpfn_reference.py \
  --dataset "$DATASET" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --device cpu \
  --workers "$SLURM_CPUS_PER_TASK" \
  --output "../outputs/real_data/tabpfn_${DATASET}.csv"
