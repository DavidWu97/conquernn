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
from real_data_metrics import pointwise_pinball, write_table2_outputs
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
    parser.add_argument(
        "--pointwise-output",
        type=Path,
        help="Per-test-point predictions and losses; defaults beside --output.",
    )
    parser.add_argument(
        "--inference-output-dir",
        type=Path,
        help="Table 2 bootstrap outputs; defaults beside --output.",
    )
    parser.add_argument("--bootstrap-chunk-size", type=int, default=100)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def predict(model, features: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict(features)).reshape(len(features), -1)[:, 0]


def pinball(target: np.ndarray, prediction: np.ndarray, quantile: float) -> float:
    return float(np.mean(pointwise_pinball(target, prediction, quantile)))


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


def result_frames(
    dataset: str,
    method: str,
    quantile: float,
    bandwidth: float,
    target: np.ndarray,
    prediction: np.ndarray,
) -> tuple[dict, pd.DataFrame]:
    if dataset.startswith("bmi_"):
        target_for_loss = np.asarray(target, dtype=np.float32)
        prediction_for_loss = np.asarray(prediction, dtype=np.float32)
        residual = target_for_loss - prediction_for_loss
        level = np.float32(quantile)
        losses = np.maximum(level * residual, (level - np.float32(1.0)) * residual)
    else:
        losses = pointwise_pinball(target, prediction, quantile)
    summary = {
        "dataset": dataset,
        "method": method,
        "quantile": quantile,
        "bandwidth": bandwidth,
        "n_test": len(target),
        "pinball_loss": float(np.mean(losses)),
    }
    pointwise = pd.DataFrame(
        {
            "dataset": dataset,
            "test_row": np.arange(len(target), dtype=int),
            "quantile": quantile,
            "target": target,
            "method": method,
            "bandwidth": bandwidth,
            "prediction": prediction,
            "pinball_loss": losses,
        }
    )
    return summary, pointwise


def run_dataset(
    dataset: str, args: argparse.Namespace
) -> tuple[list[dict], list[pd.DataFrame]]:
    x_train, x_test, y_train, y_test = paper_split(dataset, args.seed)
    rows = []
    pointwise_frames = []
    for quantile in QUANTILES:
        baseline = QuantileNetwork(quantiles=quantile, shape=SHAPE, residual=False)
        baseline.fit(x_train, y_train, stop=False)
        summary, pointwise = result_frames(
            dataset,
            "Nonsmooth QRNN",
            quantile,
            np.nan,
            y_test,
            predict(baseline, x_test),
        )
        rows.append(summary)
        pointwise_frames.append(pointwise)
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
            summary, pointwise = result_frames(
                dataset,
                f"ConquerNet--{kernel.title()}",
                quantile,
                bandwidth,
                y_test,
                predict(model, x_test),
            )
            rows.append(summary)
            pointwise_frames.append(pointwise)
    return rows, pointwise_frames


def main() -> None:
    args = parse_args()
    if args.bootstrap_chunk_size < 1:
        raise ValueError("--bootstrap-chunk-size must be positive")
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    rows = []
    pointwise_frames = []
    bmi_started = False
    for dataset in datasets:
        if dataset.startswith("bmi_") and not bmi_started:
            set_seed(args.seed)
            bmi_started = True
        elif dataset == "california_housing":
            set_seed(args.seed)
        dataset_rows, dataset_pointwise = run_dataset(dataset, args)
        rows.extend(dataset_rows)
        pointwise_frames.extend(dataset_pointwise)
    output = pd.DataFrame(rows)
    pointwise = pd.concat(pointwise_frames, ignore_index=True)
    pointwise_output = args.pointwise_output or args.output.with_name(
        f"{args.output.stem}_pointwise.csv.gz"
    )
    inference_output = args.inference_output_dir or args.output.parent / (
        f"{args.output.stem}_table2_inference"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    pointwise_output.parent.mkdir(parents=True, exist_ok=True)
    pointwise.to_csv(pointwise_output, index=False, compression="gzip")
    pointwise_inference, pooled_inference = write_table2_outputs(
        pointwise,
        inference_output,
        chunk_size=args.bootstrap_chunk_size,
    )
    print(
        output.pivot(
            index=["dataset", "method"], columns="quantile", values="pinball_loss"
        )
    )
    print(f"Saved {args.output}")
    print(f"Saved {pointwise_output}")
    print(pointwise_inference.to_string(index=False))
    print(pooled_inference.to_string(index=False))
    print(f"Saved Table 2 inference files under {inference_output}")


if __name__ == "__main__":
    main()
