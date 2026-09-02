# ConquerNet

This repository reproduces the experiments in **ConquerNet: Convolution-Smoothed
Quantile ReLU Neural Networks with Minimax Guarantees**. Simulation and real-data
workflows are separated, while the neural-network implementation shared by both
experiments remains at the root of `python/`.

## Repository layout

```text
conquernn/
├── python/
│   ├── baseline.py                 # Nonsmoothed quantile network
│   ├── conquer_model.py            # Convolution-smoothed quantile network
│   ├── loss.py
│   ├── torch_utils.py
│   ├── utils.py
│   ├── simulation/
│   │   ├── main.py                 # Main simulation study
│   │   ├── additional_baselines.py # QRF and Huber appendix baselines
│   │   ├── scenarios.py
│   │   ├── tree_baselines.py
│   │   └── summarize.py
│   └── real_data/
│       ├── bootstrap.py            # Default fixed-model Table 2 workflow
│       ├── metrics.py              # Paired-bootstrap inference
│       ├── train.py                # Optional training-from-scratch workflow
│       ├── tabpfn.py               # TabPFN-3 reference
│       ├── data/                   # Original BMI and Housing inputs
│       └── models/                 # 60 camera-ready neural checkpoints
├── jobs/
│   ├── simulation/
│   └── real_data/
├── environment.yml
├── requirements.txt
└── requirements-tabpfn.txt
```

Generated predictions, result tables, and logs are excluded from version control.
The fixed neural checkpoints needed for Table 2 are included in the repository.

## Environment

The reference environment uses Python 3.10.12, NumPy 1.26.4, pandas 2.2.3,
SciPy 1.15.3, scikit-learn 1.7.2, PyTorch 2.6.0, and quantile-forest 1.4.1.

```bash
conda env create -f environment.yml
conda activate conquernet
```

Alternatively:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Real-data Table 2: fixed-model reproduction

This is the recommended and fastest reproduction path. It does **not** retrain
or select hyperparameters. It verifies and loads the 60 fitted models in
`python/real_data/models/`, recreates the fixed seed-42 test split, generates
held-out predictions, and performs the paired bootstrap reported in the paper.

```bash
cd python
python -m real_data.bootstrap
```

The same command is available from the repository root:

```bash
jobs/real_data/bootstrap_saved_models.sh
```

The command writes:

- `outputs/real_data/neural_results.csv`: the 60 neural pinball-loss cells;
- `outputs/real_data/neural_results_pointwise.csv.gz`: temporary row-level
  predictions and losses used by the bootstrap;
- `outputs/real_data/table2_inference/table2_pointwise_inference.csv`: 45
  ConquerNet-versus-nonsmoothed comparisons and pointwise 95% basic intervals;
- `outputs/real_data/table2_inference/table2_pooled_inference.csv`: the pooled
  BMI and California Housing estimates, confidence intervals, and p-values;
- `outputs/real_data/saved_model_metadata.json`: split, environment, checkpoint,
  and SHA-256 metadata.

All checkpoint hashes are validated against
`python/real_data/models/manifest.json` before `torch.load` is called. The bundled
checkpoints and pinned environment were locally verified against the camera-ready
paper: all 60 neural cells, all 45 pointwise inference rows, and both pooled rows
match at the precision displayed in the paper.

### Bootstrap estimands

Each bootstrap sample resamples held-out subjects and keeps the nonsmoothed and
ConquerNet losses paired. The reported difference is
`Nonsmooth QRNN loss - ConquerNet loss`.

- BMI pointwise intervals use 20,000 draws with seeds 20260718 (male) and
  20260719 (female).
- California Housing pointwise intervals use 100,000 draws with seed 20260719.
- The pooled BMI analysis uses 100,000 draws with seed 20260721 and the centered
  absolute two-sided p-value.
- The pooled Housing analysis uses 100,000 draws with seed 20260720 and the
  equal-tail p-value that inverts the basic interval.

These intervals quantify held-out-sample uncertainty conditional on the fixed
fitted models and split. They intentionally do not include retraining variability.

The camera-ready pooled results are:

| Estimand | Improvement | 95% basic CI | p-value |
|---|---:|---:|---:|
| BMI | 0.002280 | [0.001059, 0.003495] | 0.00029 |
| California Housing | 0.004425 | [0.003419, 0.005413] | 0.00002 |

### Camera-ready pinball losses

| Dataset | Method | q=.05 | q=.25 | q=.50 | q=.75 | q=.95 |
|---|---|---:|---:|---:|---:|---:|
| BMI male | Nonsmooth QRNN | 0.5231 | 2.0638 | 2.7387 | 2.0534 | 0.5190 |
| BMI male | TabPFN-3 | 0.5214 | 2.0611 | 2.7380 | 2.0513 | 0.5191 |
| BMI male | ConquerNet--Gaussian | 0.5221 | 2.0609 | 2.7384 | 2.0513 | 0.5189 |
| BMI male | ConquerNet--Uniform | 0.5217 | 2.0618 | 2.7403 | 2.0521 | 0.5192 |
| BMI male | ConquerNet--Epanechnikov | 0.5218 | 2.0636 | 2.7389 | 2.0518 | 0.5205 |
| BMI female | Nonsmooth QRNN | 0.5218 | 2.0596 | 2.7366 | 2.0555 | 0.5249 |
| BMI female | TabPFN-3 | 0.5202 | 2.0446 | 2.7286 | 2.0522 | 0.5248 |
| BMI female | ConquerNet--Gaussian | 0.5202 | 2.0446 | 2.7323 | 2.0531 | 0.5259 |
| BMI female | ConquerNet--Uniform | 0.5211 | 2.0497 | 2.7311 | 2.0642 | 0.5251 |
| BMI female | ConquerNet--Epanechnikov | 0.5205 | 2.0454 | 2.7276 | 2.0523 | 0.5248 |
| California Housing | Nonsmooth QRNN | 0.0426 | 0.1418 | 0.1879 | 0.1698 | 0.0671 |
| California Housing | TabPFN-3 | 0.0272 | 0.0843 | 0.1143 | 0.1013 | 0.0397 |
| California Housing | ConquerNet--Gaussian | 0.0436 | 0.1336 | 0.1823 | 0.1630 | 0.0645 |
| California Housing | ConquerNet--Uniform | 0.0434 | 0.1327 | 0.1815 | 0.1590 | 0.0680 |
| California Housing | ConquerNet--Epanechnikov | 0.0438 | 0.1391 | 0.1782 | 0.1609 | 0.0676 |

## Optional real-data training from scratch

Use this only to repeat model fitting and five-fold bandwidth selection. It is
substantially slower and is not required to reproduce the conditional bootstrap.

```bash
cd python
python -m real_data.train \
  --dataset all \
  --output ../outputs/real_data/retrained_results.csv \
  --pointwise-output ../outputs/real_data/retrained_pointwise.csv.gz \
  --inference-output-dir ../outputs/real_data/retrained_table2_inference
```

Training is stochastic, so small differences may occur across hardware and
library versions. The fixed-model command above is the exact paper-result path.

## Main simulation

The main protocol contains 50 repetitions for six scenario and architecture
configurations and is best run on a compute cluster.

```bash
cd python
python -m simulation.main \
  --scenario 1 \
  --shape small \
  --trials 50 \
  --workers 48 \
  --output ../outputs/simulation/scenario1_small.npz
```

Use `--shape small` for `(5, 70)` and `--shape large` for `(10, 50)`. Repeat for
scenarios 1, 2, and 3 and both shapes. Summarize an output file with:

```bash
python -m simulation.summarize \
  ../outputs/simulation/scenario1_small.npz \
  --output ../outputs/simulation/scenario1_small.csv
```

Appendix Table 18 adds Quantile Regression Forest and Huberized neural baselines:

```bash
python -m simulation.additional_baselines \
  --scenario 1 \
  --shape small \
  --trials 50 \
  --workers 48 \
  --output ../outputs/additional_baselines/scenario1_small.npz
```

The QRF protocol uses 300 trees, `min_samples_split=10`,
`min_samples_leaf=5`, and `random_state=0`. The Huber loss uses
`huber_delta=0.5`.

## TabPFN-3 reference

TabPFN is optional and is not needed for the fixed-model ConquerNet bootstrap.
The camera-ready reference used `tabpfn==8.5.0`, ModelVersion V3,
`n_estimators=8`, and checkpoint
`tabpfn-v3-regressor-v3_default.ckpt` with SHA-256
`311ce18d97e9533d8585eaadafe040fbdd8070533209ed8696641dadc97a7301`.

```bash
python -m pip install -r requirements-tabpfn.txt
cd python
python -m real_data.tabpfn \
  --dataset all \
  --checkpoint-dir /path/to/tabpfn-cache \
  --device cpu \
  --workers 48 \
  --output ../outputs/real_data/tabpfn_results.csv
```

## Slurm

```bash
export PROJECT_DIR=/absolute/path/to/conquernn
sbatch jobs/simulation/main_array.sh
sbatch jobs/simulation/additional_baselines_array.sh
sbatch jobs/real_data/train_from_scratch.sh

export CHECKPOINT_DIR=/absolute/path/to/tabpfn-cache
sbatch jobs/real_data/tabpfn_array.sh
```

On ParaCloud the job scripts load `gcc/12.2` from
`/public1/soft/modules/module.sh`. Override `MODULE_INIT`, `GCC_MODULE`, or
`PYTHON_BIN` for another cluster.

## Data and privacy

The repository contains the original source datasets required by the scripts.
Generated row-level BMI predictions are ignored by Git and must not be committed.
The bundled checkpoints contain fitted parameters only; no held-out prediction
rows or bootstrap resamples are stored in the repository.
