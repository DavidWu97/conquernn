# ConquerNet

This repository contains the implementation and original real-data inputs for **ConquerNet: Convolution-Smoothed Quantile ReLU Neural Networks with Minimax Guarantees**. It includes the main simulation study, the BMI and California Housing experiments, and the TabPFN-3 reference used in the camera-ready paper.

## Repository layout

```text
conquernn/
├── README.md
├── requirements.txt
├── requirements-tabpfn.txt
├── environment.yml
├── jobs/                       # Portable Slurm entry points
└── python/
    ├── baseline.py             # Nonsmoothed quantile neural network
    ├── conquer_model.py        # Convolution-smoothed quantile network
    ├── loss.py                 # Smoothed and pinball losses
    ├── scenario.py             # Simulation data-generating processes
    ├── simulation_benchmark.py
    ├── summarize_simulation.py
    ├── real_data.py
    ├── real_data_benchmark.py
    ├── tabpfn_reference.py
    └── data/                   # Original BMI and California Housing inputs
```

Generated models, predictions, tables, logs, and experiment outputs are excluded from version control.

## Environment

The experiments use Python 3.10. The pinned environment includes NumPy 1.26.4, pandas 2.2.3, SciPy 1.15.3, scikit-learn 1.7.2, and PyTorch 2.6.0.

### Conda

```bash
conda env create -f environment.yml
conda activate conquernet
```

### venv

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For a CPU-only installation, install the matching PyTorch wheel before the remaining requirements:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Paper-scale experiments are CPU-intensive. The full main simulation protocol contains 50 repetitions for six scenario and network-shape configurations and is best submitted to a compute cluster.

## Main simulation experiment

The main simulation compares the nonsmoothed QRNN with Gaussian, uniform, and Epanechnikov ConquerNet models at five quantile levels and three sample sizes. Run each scenario and network shape separately:

```bash
cd python
python simulation_benchmark.py \
  --scenario 1 \
  --shape small \
  --trials 50 \
  --workers 48 \
  --output ../outputs/simulation/scenario1_small.npz
```

Use `--shape small` for `(5, 70)` and `--shape large` for `(10, 50)`. Repeat the command for scenarios 1, 2, and 3 and both shapes. The Slurm array in `jobs/simulation_array.sh` submits all six configurations.

Convert a generated NPZ file to a tidy CSV table:

```bash
python summarize_simulation.py \
  ../outputs/simulation/scenario1_small.npz \
  --output ../outputs/simulation/scenario1_small.csv
```

Useful protocol variants are:

```bash
# Train ConquerNet for all epochs
python simulation_benchmark.py --scenario 1 --shape small --no-stop \
  --output ../outputs/simulation/scenario1_small_no_stop.npz

# Residual-network experiment
python simulation_benchmark.py --scenario 1 --shape small --residual \
  --output ../outputs/simulation/scenario1_small_residual.npz

# Joint noncrossing quantiles
python simulation_benchmark.py --scenario 1 --shape small --mode joint \
  --output ../outputs/simulation/scenario1_small_joint.npz
```

## Real-data experiment

The original inputs are included under `python/data/`:

- `data/bmi/health_lifestyle_dataset.csv`: the [Health and Lifestyle dataset](https://www.kaggle.com/datasets/chik0di/health-and-lifestyle-dataset/). The experiment keeps observations with `family_history == 0`, analyzes male and female observations separately, and predicts BMI from age, daily steps, sleep hours, water intake, and calories consumed.
- `data/housing/cal_housing.tgz`: the original California Housing archive. `real_data.py` applies the same column reordering and ratio transformations used by scikit-learn.

The experiments use a fixed 80/20 split with seed 42, network shape `(10, 50)`, and quantiles `(0.05, 0.25, 0.50, 0.75, 0.95)`. Run the complete neural-network protocol, including five-fold bandwidth selection on the training set:

```bash
cd python
python real_data_benchmark.py \
  --dataset all \
  --output ../outputs/real_data/neural_results.csv
```

Training is stochastic and results may vary slightly with library versions and hardware.

## TabPFN-3 reference

The camera-ready experiment uses TabPFN-3 as an external reference on the same three datasets, split, and quantile levels. Its configuration is:

- `tabpfn==8.5.0` and `ModelVersion.V3`;
- checkpoint `tabpfn-v3-regressor-v3_default.ckpt`;
- checkpoint SHA-256 `311ce18d97e9533d8585eaadafe040fbdd8070533209ed8696641dadc97a7301`;
- `n_estimators=8`;
- native TabPFN preprocessing and quantile output;
- no task-specific tuning, calibration, or quantile postprocessing.

Install the optional dependency:

```bash
python -m pip install -r requirements-tabpfn.txt
```

Place the checkpoint in a local cache directory, then run:

```bash
cd python
python tabpfn_reference.py \
  --dataset all \
  --checkpoint-dir /path/to/tabpfn-cache \
  --device cpu \
  --workers 48 \
  --output ../outputs/real_data/tabpfn_results.csv
```

The script checks the TabPFN package version and checkpoint hash before fitting. `jobs/tabpfn_array.sh` provides a three-dataset Slurm array.

## Slurm

The Slurm scripts use environment variables for project and Python paths:

```bash
export PROJECT_DIR=/absolute/path/to/conquernn
sbatch jobs/simulation_array.sh
sbatch jobs/real_data.sh

export CHECKPOINT_DIR=/absolute/path/to/tabpfn-cache
sbatch jobs/tabpfn_array.sh
```

On ParaCloud, the scripts load `gcc/12.2` from `/public1/soft/modules/module.sh`. On another cluster, set `MODULE_INIT` and `GCC_MODULE`, or leave `MODULE_INIT` pointing to a nonexistent file when no compiler runtime module is needed.

Output files are written under `outputs/`, which is ignored by Git.

## Reproducibility notes

- Use Python 3.10 and the pinned package versions for the closest numerical reproduction.
- The simulation runner preserves the data sequence, initialization seed, nonsmoothed training rule, and ConquerNet early-stopping rule used in the paper.
- The paper reports sample standard deviations across 50 repetitions.
- Runtime values depend on hardware and system load.
- The TabPFN checkpoint is not tracked by Git. Do not commit access tokens, model caches, or row-level BMI predictions.
