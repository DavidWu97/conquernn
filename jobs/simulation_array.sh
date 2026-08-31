#!/bin/bash
#SBATCH --job-name=conquernet-sim
#SBATCH --array=0-5
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=outputs/logs/simulation_%A_%a.out
#SBATCH --error=outputs/logs/simulation_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODULE_INIT="${MODULE_INIT:-/public1/soft/modules/module.sh}"
SCENARIOS=(1 1 2 2 3 3)
SHAPES=(small large small large small large)
SCENARIO="${SCENARIOS[$SLURM_ARRAY_TASK_ID]}"
SHAPE="${SHAPES[$SLURM_ARRAY_TASK_ID]}"

if [[ -f "$MODULE_INIT" ]]; then
  source "$MODULE_INIT"
  module load "${GCC_MODULE:-gcc/12.2}"
fi

mkdir -p "$PROJECT_DIR/outputs/logs" "$PROJECT_DIR/outputs/simulation"
cd "$PROJECT_DIR/python"
"$PYTHON_BIN" -u simulation_benchmark.py \
  --scenario "$SCENARIO" \
  --shape "$SHAPE" \
  --trials 50 \
  --workers "$SLURM_CPUS_PER_TASK" \
  --output "../outputs/simulation/scenario${SCENARIO}_${SHAPE}.npz"
