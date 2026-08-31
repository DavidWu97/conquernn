"""Run the QRF and Huber baselines reported in Appendix Table 18."""

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
from baseline_huber import HuberQuantileNetwork
from scenario import Scenario1, Scenario2, Scenario3
from tree_baselines import QRF

METHODS = ("qrf", "huber")
SCENARIOS = {1: Scenario1, 2: Scenario2, 3: Scenario3}
SHAPES = {"small": (5, 70), "large": (10, 50)}
_DATA = None
_CONFIG = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=int, choices=SCENARIOS, required=True)
    parser.add_argument("--shape", choices=SHAPES, required=True)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--test-size", type=int, default=10000)
    parser.add_argument(
        "--sample-sizes", type=int, nargs="+", default=[1000, 5000, 10000]
    )
    parser.add_argument(
        "--quantiles", type=float, nargs="+", default=[0.05, 0.25, 0.5, 0.75, 0.95]
    )
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--huber-delta", type=float, default=0.5)
    parser.add_argument("--qrf-trees", type=int, default=300)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def make_data(config: dict) -> list[dict]:
    np.random.seed(config["seed"])
    scenario = SCENARIOS[config["scenario"]]()
    levels = np.asarray(config["quantiles"])
    trials = []
    for _ in range(config["trials"]):
        test_x = np.random.random((config["test_size"], scenario.n_in))
        test_y = scenario.sample(test_x)
        truth = np.column_stack([scenario.quantile(test_x, q) for q in levels])
        training = []
        for sample_size in config["sample_sizes"]:
            train_x = np.random.random((sample_size, scenario.n_in))
            training.append((train_x, scenario.sample(train_x)))
        trials.append(
            {"test_x": test_x, "test_y": test_y, "truth": truth, "training": training}
        )
    return trials


def initialize_worker(data: list[dict], config: dict) -> None:
    global _DATA, _CONFIG
    _DATA, _CONFIG = data, config
    torch.set_num_threads(1)


def fit_one(task: tuple[int, int, int, int]) -> tuple:
    trial, sample_index, method_index, quantile_index = task
    item = _DATA[trial]
    train_x, train_y = item["training"][sample_index]
    method = _CONFIG["methods"][method_index]
    quantile = float(_CONFIG["quantiles"][quantile_index])
    np.random.seed(_CONFIG["seed"])
    torch.manual_seed(_CONFIG["seed"])
    warnings.filterwarnings("ignore", message=r"The epoch parameter in `scheduler.step")

    started = time.perf_counter()
    if method == "qrf":
        model = QRF(
            quantiles=quantile,
            n_estimators=_CONFIG["qrf_trees"],
            min_samples_split=10,
            min_samples_leaf=5,
            n_jobs=1,
            random_state=0,
            val_pct=0.1,
        )
        model.fit(train_x, train_y)
    else:
        model = HuberQuantileNetwork(
            quantiles=quantile,
            shape=tuple(_CONFIG["shape_value"]),
            huber_delta=_CONFIG["huber_delta"],
            residual=False,
        )
        model.fit(train_x, train_y, stop=True)
    elapsed = time.perf_counter() - started

    prediction = np.asarray(model.predict(item["test_x"])).reshape(-1)
    truth = item["truth"][:, quantile_index]
    mse = float(np.mean((truth - prediction) ** 2))
    return trial, method_index, sample_index, quantile_index, mse, elapsed


def collect(config: dict, data: list[dict]) -> dict[str, np.ndarray]:
    shape = (
        config["trials"],
        len(config["methods"]),
        len(config["sample_sizes"]),
        len(config["quantiles"]),
    )
    mse = np.full(shape, np.nan)
    training_time = np.full(shape, np.nan)
    tasks = [
        (trial, sample_index, method_index, quantile_index)
        for trial in range(config["trials"])
        for sample_index in range(len(config["sample_sizes"]))
        for quantile_index in range(len(config["quantiles"]))
        for method_index in range(len(config["methods"]))
    ]

    if config["workers"] == 1:
        initialize_worker(data, config)
        fitted = map(fit_one, tasks)
        pool = None
    else:
        pool = mp.get_context("spawn").Pool(
            config["workers"], initializer=initialize_worker, initargs=(data, config)
        )
        fitted = pool.imap_unordered(fit_one, tasks)

    try:
        for trial, method, sample, quantile, value, elapsed in fitted:
            mse[trial, method, sample, quantile] = value
            training_time[trial, method, sample, quantile] = elapsed
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return {"mse": mse, "training_time": training_time}


def main() -> None:
    args = parse_args()
    config = vars(args).copy()
    config["shape_value"] = SHAPES[config["shape"]]
    config["output"] = str(config["output"])
    if config["trials"] < 1 or config["workers"] < 1:
        raise ValueError("trials and workers must be positive")
    print(json.dumps(config, indent=2))
    arrays = collect(config, make_data(config))
    output = Path(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        **arrays,
        scenario=config["scenario"],
        shape=np.asarray(config["shape_value"]),
        shape_label=config["shape"],
        trials=config["trials"],
        sample_sizes=np.asarray(config["sample_sizes"]),
        quantiles=np.asarray(config["quantiles"]),
        methods=np.asarray(config["methods"]),
        huber_delta=config["huber_delta"],
        qrf_trees=config["qrf_trees"],
        seed=config["seed"],
    )
    means = np.nanmean(arrays["mse"], axis=0)
    for method, table in zip(config["methods"], means):
        print(f"{method} mean MSE")
        print(np.array2string(table, precision=4, floatmode="fixed"))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
