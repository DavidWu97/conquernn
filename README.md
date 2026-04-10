# ConquerNet

Codes for paper **ConquerNet: Convolution-Smoothed Quantile ReLU Neural Networks with Minimax Guarantees** submitted to TMLR.

Our conquer method is implemented in `python/conquer_model.py`. Run `python/benchmark.py` to get tables for MSEs of a single quantile level and running times. Run `python/benchmark_multiquantile.py` to get tables for MSEs of multiple quantile levels and running times. Run `python/benchmark_CV.py` to get tables for MSEs and running times by K-fold cross-validation. Run `python/bmi_analysis.py` to get tables for pinball losses of a real BMI dataset. Run `python/visualize.py` to get figures for running times and loss curves.

## Requirements

`python3` is required. Related packages required: `pytorch`, `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`.
