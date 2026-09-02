"""Reproduce the TabPFN-3 rows added to camera-ready Table 2."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import os
from pathlib import Path

import numpy as np
import pandas as pd
from real_data.datasets import DATASETS, paper_split

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
VERSION = "8.5.0"
CHECKPOINT = "tabpfn-v3-regressor-v3_default.ckpt"
CHECKPOINT_SHA256 = "311ce18d97e9533d8585eaadafe040fbdd8070533209ed8696641dadc97a7301"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("all", *DATASETS), default="all")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(dataset: str, repetition: int = 0) -> int:
    payload = f"{dataset}|{repetition}|TabPFN-3|final".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 2_147_483_647


def normalize_prediction(raw, n_test: int) -> np.ndarray:
    if isinstance(raw, (list, tuple)):
        prediction = np.column_stack([np.asarray(item).reshape(-1) for item in raw])
    else:
        prediction = np.asarray(raw)
        if prediction.shape == (len(QUANTILES), n_test):
            prediction = prediction.T
        prediction = prediction.reshape(n_test, len(QUANTILES))
    if (
        prediction.shape != (n_test, len(QUANTILES))
        or not np.isfinite(prediction).all()
    ):
        raise ValueError(f"Invalid TabPFN prediction array: {prediction.shape}")
    return prediction.astype(np.float64, copy=False)


def pinball(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    residual = target[:, None] - prediction
    levels = np.asarray(QUANTILES)[None, :]
    return np.mean(np.maximum(levels * residual, (levels - 1.0) * residual), axis=0)


def check_installation(args: argparse.Namespace) -> None:
    installed = importlib.metadata.version("tabpfn")
    if installed != VERSION:
        raise RuntimeError(f"Expected tabpfn=={VERSION}, found {installed}")
    checkpoint = args.checkpoint_dir / CHECKPOINT
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    observed = sha256(checkpoint)
    if observed != CHECKPOINT_SHA256:
        raise RuntimeError(f"Checkpoint SHA-256 mismatch: {observed}")


def run_dataset(name: str, args: argparse.Namespace) -> list[dict]:
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    x_train, x_test, y_train, y_test = paper_split(name, args.seed)
    model = TabPFNRegressor.create_default_for_version(
        ModelVersion.V3,
        n_estimators=args.n_estimators,
        device=args.device,
        random_state=stable_seed(name),
        fit_mode="fit_preprocessors",
        memory_saving_mode=True,
        n_preprocessing_jobs=args.workers,
        tuning_config=None,
        show_progress_bar=True,
    )
    model.fit(x_train, y_train)
    raw = model.predict(x_test, output_type="quantiles", quantiles=list(QUANTILES))
    losses = pinball(y_test, normalize_prediction(raw, len(y_test)))
    return [
        {
            "dataset": name,
            "method": "TabPFN-3",
            "quantile": quantile,
            "n_test": len(y_test),
            "pinball_loss": loss,
        }
        for quantile, loss in zip(QUANTILES, losses)
    ]


def main() -> None:
    args = parse_args()
    os.environ["TABPFN_MODEL_CACHE_DIR"] = str(args.checkpoint_dir.resolve())
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    if args.device == "cpu":
        os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "true")
    check_installation(args)
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    table = pd.DataFrame(
        [row for dataset in datasets for row in run_dataset(dataset, args)]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(table.pivot(index="dataset", columns="quantile", values="pinball_loss"))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
