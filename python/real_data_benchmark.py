"""Train the paper's neural methods on BMI and California Housing."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from baseline import QuantileNetwork
from conquer_model import ConquerNetwork
from real_data import DATASETS, paper_split
from sklearn.model_selection import KFold

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
KERNELS = ("gaussian", "uniform", "epanechnikov")
BANDWIDTHS = (0.001, 0.005, 0.01, 0.05, 0.1)
SHAPE = (10, 50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("all", *DATASETS), default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def predict(model, features: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict(features)).reshape(len(features), -1)[:, 0]


def pinball(target: np.ndarray, prediction: np.ndarray, quantile: float) -> float:
    residual = target - prediction
    return float(np.mean(np.maximum(quantile * residual, (quantile - 1.0) * residual)))


def select_bandwidth(
    x_train: np.ndarray,
    y_train: np.ndarray,
    quantile: float,
    kernel: str,
    folds: int,
) -> float:
    splitter = KFold(n_splits=folds, shuffle=True, random_state=42)
    losses = {bandwidth: [] for bandwidth in BANDWIDTHS}
    for bandwidth in BANDWIDTHS:
        for train_index, validation_index in splitter.split(x_train):
            model = ConquerNetwork(
                quantiles=quantile,
                kernel=kernel,
                bandwidth=bandwidth,
                shape=SHAPE,
                residual=False,
            )
            model.fit(x_train[train_index], y_train[train_index])
            value = pinball(
                y_train[validation_index],
                predict(model, x_train[validation_index]),
                quantile,
            )
            losses[bandwidth].append(value)
    return min(losses, key=lambda bandwidth: np.mean(losses[bandwidth]))


def run_dataset(dataset: str, args: argparse.Namespace) -> list[dict]:
    x_train, x_test, y_train, y_test = paper_split(dataset, args.seed)
    rows = []
    for quantile in QUANTILES:
        baseline = QuantileNetwork(quantiles=quantile, shape=SHAPE, residual=False)
        baseline.fit(x_train, y_train, stop=False)
        rows.append(
            {
                "dataset": dataset,
                "method": "Nonsmooth QRNN",
                "quantile": quantile,
                "bandwidth": np.nan,
                "n_test": len(y_test),
                "pinball_loss": pinball(y_test, predict(baseline, x_test), quantile),
            }
        )
        for kernel in KERNELS:
            bandwidth = select_bandwidth(
                x_train, y_train, quantile, kernel, args.cv_folds
            )
            model = ConquerNetwork(
                quantiles=quantile,
                kernel=kernel,
                bandwidth=bandwidth,
                shape=SHAPE,
                residual=False,
            )
            model.fit(x_train, y_train)
            rows.append(
                {
                    "dataset": dataset,
                    "method": f"ConquerNet--{kernel.title()}",
                    "quantile": quantile,
                    "bandwidth": bandwidth,
                    "n_test": len(y_test),
                    "pinball_loss": pinball(y_test, predict(model, x_test), quantile),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    rows = []
    bmi_started = False
    for dataset in datasets:
        if dataset.startswith("bmi_") and not bmi_started:
            set_seed(args.seed)
            bmi_started = True
        elif dataset == "california_housing":
            set_seed(args.seed)
        rows.extend(run_dataset(dataset, args))
    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(
        output.pivot(
            index=["dataset", "method"], columns="quantile", values="pinball_loss"
        )
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
