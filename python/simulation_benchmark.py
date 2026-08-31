"""Run the simulation protocol used for the main and appendix tables."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from baseline import QuantileNetwork
from conquer_model import ConquerNetwork
from scenario import Scenario1, Scenario2, Scenario3
from utils import get_idx

METHODS = ("baseline", "gaussian", "uniform", "epanechnikov")
SCENARIOS = {1: Scenario1, 2: Scenario2, 3: Scenario3}
SHAPES = {"small": (5, 70), "large": (10, 50)}
_DATA = None
_CONFIG = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=int, choices=SCENARIOS, required=True)
    parser.add_argument("--shape", choices=SHAPES, required=True)
    parser.add_argument("--mode", choices=("single", "joint"), default="single")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--test-size", type=int, default=10000)
    parser.add_argument(
        "--sample-sizes", type=int, nargs="+", default=[1000, 5000, 10000]
    )
    parser.add_argument(
        "--quantiles", type=float, nargs="+", default=[0.05, 0.25, 0.5, 0.75, 0.95]
    )
    parser.add_argument(
        "--bandwidths", type=float, nargs="+", default=[0.001, 0.005, 0.01, 0.05, 0.1]
    )
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--residual", action="store_true")
    parser.add_argument(
        "--manual-gradient", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def make_data(config: dict) -> list[dict]:
    np.random.seed(config["seed"])
    scenario = SCENARIOS[config["scenario"]]()
    quantiles = np.asarray(config["quantiles"])
    trials = []
    for _ in range(config["trials"]):
        x_test = np.random.random((config["test_size"], scenario.n_in))
        y_test = scenario.sample(x_test)
        truth = np.column_stack([scenario.quantile(x_test, q) for q in quantiles])
        training = []
        for sample_size in config["sample_sizes"]:
            x_train = np.random.random((sample_size, scenario.n_in))
            training.append((x_train, scenario.sample(x_train)))
        trials.append(
            {"x_test": x_test, "y_test": y_test, "truth": truth, "training": training}
        )
    return trials


def initialize_worker(data: list[dict], config: dict) -> None:
    global _DATA, _CONFIG
    _DATA = data
    _CONFIG = config
    torch.set_num_threads(1)


def fit_one(task: tuple[int, int, int, int]) -> tuple:
    trial, sample_index, method_index, quantile_index = task
    method = METHODS[method_index]
    warnings.filterwarnings(
        "ignore",
        message=r"The epoch parameter in `scheduler.step\(\)`",
        category=UserWarning,
    )
    item = _DATA[trial]
    x_train, y_train = item["training"][sample_index]
    quantiles = np.asarray(_CONFIG["quantiles"])
    bandwidths = np.asarray(_CONFIG["bandwidths"])
    shape = tuple(_CONFIG["shape_value"])

    if _CONFIG["mode"] == "joint":
        model_quantiles = quantiles
    else:
        model_quantiles = float(quantiles[quantile_index])

    bandwidth_indices = [0] if method == "baseline" else range(len(bandwidths))
    results = []
    for bandwidth_index in bandwidth_indices:
        np.random.seed(_CONFIG["seed"])
        torch.manual_seed(_CONFIG["seed"])
        started = time.perf_counter()
        if method == "baseline":
            model = QuantileNetwork(
                quantiles=model_quantiles, shape=shape, residual=_CONFIG["residual"]
            )
            model.fit(x_train, y_train, stop=False)
        else:
            model = ConquerNetwork(
                quantiles=model_quantiles,
                kernel=method,
                bandwidth=float(bandwidths[bandwidth_index]),
                shape=shape,
                residual=_CONFIG["residual"],
            )
            model.fit(
                x_train,
                y_train,
                manual_grad=_CONFIG["manual_gradient"],
                stop=_CONFIG["stop"],
            )
        elapsed = time.perf_counter() - started
        prediction = np.asarray(model.predict(item["x_test"])).reshape(
            len(item["x_test"]), -1
        )
        if _CONFIG["mode"] == "single":
            prediction = prediction[:, :1]
            truth = item["truth"][:, quantile_index : quantile_index + 1]
            observed = item["y_test"][:, None]
            levels = quantiles[quantile_index : quantile_index + 1]
        else:
            truth = item["truth"]
            observed = item["y_test"][:, None]
            levels = quantiles
        mse = np.mean((truth - prediction) ** 2, axis=0)
        mae = np.mean(np.abs(truth - prediction), axis=0)
        coverage_bias = np.abs(np.mean(observed <= prediction, axis=0) - levels)
        if _CONFIG["mode"] == "single":
            mse, mae, coverage_bias = (
                float(mse[0]),
                float(mae[0]),
                float(coverage_bias[0]),
            )
        results.append((bandwidth_index, mse, mae, coverage_bias, elapsed))
    return trial, sample_index, method_index, quantile_index, results


def tasks_for(config: dict) -> list[tuple[int, int, int, int]]:
    quantile_indices = (
        range(len(config["quantiles"])) if config["mode"] == "single" else [0]
    )
    return [
        (trial, sample_index, method_index, quantile_index)
        for trial in range(config["trials"])
        for sample_index in range(len(config["sample_sizes"]))
        for quantile_index in quantile_indices
        for method_index in range(len(METHODS))
    ]


def collect(config: dict, data: list[dict]) -> dict[str, np.ndarray]:
    shape = (
        config["trials"],
        len(config["bandwidths"]),
        len(METHODS),
        len(config["sample_sizes"]),
        len(config["quantiles"]),
    )
    arrays = {
        name: np.full(shape, np.nan)
        for name in ("mse", "mae", "coverage_bias", "training_time")
    }
    tasks = tasks_for(config)
    if config["workers"] == 1:
        initialize_worker(data, config)
        fitted = map(fit_one, tasks)
    else:
        context = mp.get_context("spawn")
        pool = context.Pool(
            config["workers"], initializer=initialize_worker, initargs=(data, config)
        )
        fitted = pool.imap_unordered(fit_one, tasks)

    try:
        for trial, sample_index, method_index, quantile_index, results in fitted:
            for bandwidth_index, mse, mae, bias, elapsed in results:
                quantile_slice = (
                    slice(None) if config["mode"] == "joint" else quantile_index
                )
                bandwidth_slice = slice(None) if method_index == 0 else bandwidth_index
                index = (
                    trial,
                    bandwidth_slice,
                    method_index,
                    sample_index,
                    quantile_slice,
                )
                arrays["mse"][index] = mse
                arrays["mae"][index] = mae
                arrays["coverage_bias"][index] = bias
                arrays["training_time"][index] = elapsed
    finally:
        if config["workers"] != 1:
            pool.close()
            pool.join()
    return arrays


def print_paper_bandwidth_summary(config: dict, mse: np.ndarray) -> None:
    if config["sample_sizes"] != [1000, 5000, 10000] or len(config["bandwidths"]) != 5:
        return
    shape_index = 0 if config["shape"] == "small" else 1
    mean = np.nanmean(mse, axis=0)
    for sample_index, sample_size in enumerate(config["sample_sizes"]):
        bandwidth_index = get_idx(config["scenario"] - 1, shape_index, sample_index)
        table = mean[bandwidth_index, :, sample_index, :]
        print(f"n={sample_size}, h={config['bandwidths'][bandwidth_index]}")
        print(np.array2string(table, precision=4, floatmode="fixed"))


def main() -> None:
    args = parse_args()
    config = vars(args).copy()
    config["shape_value"] = SHAPES[config["shape"]]
    config["output"] = str(config["output"])
    if config["trials"] < 1 or config["workers"] < 1:
        raise ValueError("trials and workers must be positive")
    print(json.dumps(config, indent=2))
    data = make_data(config)
    arrays = collect(config, data)
    output = Path(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        **arrays,
        scenario=config["scenario"],
        shape=np.asarray(config["shape_value"]),
        shape_label=config["shape"],
        mode=config["mode"],
        trials=config["trials"],
        sample_sizes=np.asarray(config["sample_sizes"]),
        quantiles=np.asarray(config["quantiles"]),
        bandwidths=np.asarray(config["bandwidths"]),
        methods=np.asarray(METHODS),
        seed=config["seed"],
        stop=config["stop"],
        residual=config["residual"],
        manual_gradient=config["manual_gradient"],
    )
    print_paper_bandwidth_summary(config, arrays["mse"])
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
