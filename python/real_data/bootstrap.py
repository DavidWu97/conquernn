"""Reproduce camera-ready Table 2 inference from fixed paper checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from real_data.datasets import DATASETS, paper_split
from real_data.metrics import pointwise_pinball, write_table2_outputs

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
KERNELS = ("gaussian", "uniform", "epanechnikov")
LABELS = {
    "bmi_male": "Male",
    "bmi_female": "Female",
    "california_housing": "CaliforniaHousing",
}
METHODS = {
    "baseline": "Nonsmooth QRNN",
    **{kernel: f"ConquerNet--{kernel.title()}" for kernel in KERNELS},
}
MODULE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = MODULE_DIR.parents[1]
DEFAULT_MODEL_ROOT = MODULE_DIR / "models"
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "outputs" / "real_data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_ROOT / "neural_results.csv"
    )
    parser.add_argument(
        "--pointwise-output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "neural_results_pointwise.csv.gz",
    )
    parser.add_argument(
        "--inference-output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "table2_inference",
    )
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-chunk-size", type=int, default=100)
    args = parser.parse_args()
    if args.bootstrap_chunk_size < 1:
        parser.error("--bootstrap-chunk-size must be positive")
    return args


def load_manifest(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint manifest: {path}")
    payload = json.loads(path.read_text())
    files = payload.get("files")
    if (
        payload.get("format_version") != 1
        or payload.get("algorithm") != "sha256"
        or not isinstance(files, dict)
        or payload.get("model_count") != len(files)
    ):
        raise ValueError(f"Invalid checkpoint manifest: {path}")
    return files


def load_model(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def checkpoint_path(
    directory: Path, dataset: str, quantile: float, method: str
) -> tuple[Path, float | None]:
    prefix = f"{LABELS[dataset]}_(10, 50)_singleq{quantile:g}"
    if method == "baseline":
        path = directory / f"{prefix}_baseline"
        if not path.is_file():
            raise FileNotFoundError(path)
        return path, None

    pattern = re.compile(
        rf"^{re.escape(prefix)}_conquer_{re.escape(method)}_besth([0-9.eE+-]+)$"
    )
    matches = []
    for candidate in directory.iterdir():
        match = pattern.match(candidate.name)
        if match:
            matches.append((candidate, float(match.group(1))))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one checkpoint for {dataset}/{quantile:g}/{method}; "
            f"found {matches}"
        )
    return matches[0]


def prediction_vector(model, features: np.ndarray) -> np.ndarray:
    prediction = np.asarray(model.predict(features)).reshape(len(features), -1)[:, 0]
    if not np.isfinite(prediction).all():
        raise ValueError("Saved model produced non-finite predictions")
    return prediction


def result_frames(
    dataset: str,
    method: str,
    quantile: float,
    bandwidth: float | None,
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

    bandwidth_value = np.nan if bandwidth is None else bandwidth
    summary = {
        "dataset": dataset,
        "method": method,
        "quantile": quantile,
        "bandwidth": bandwidth_value,
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
            "bandwidth": bandwidth_value,
            "prediction": prediction,
            "pinball_loss": losses,
        }
    )
    return summary, pointwise


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    model_root = args.model_root.resolve()
    bmi_model_dir = model_root / "bmi"
    housing_model_dir = model_root / "housing"
    manifest_path = (args.manifest or model_root / "manifest.json").resolve()
    expected_hashes = load_manifest(manifest_path)
    for directory in (bmi_model_dir, housing_model_dir):
        if not directory.is_dir():
            raise FileNotFoundError(directory)

    rows = []
    pointwise_frames = []
    checkpoints = []
    for dataset in DATASETS:
        _, x_test, _, y_test = paper_split(dataset, args.seed)
        directory = (
            bmi_model_dir if dataset.startswith("bmi_") else housing_model_dir
        )
        for quantile in QUANTILES:
            for method_key, method_name in METHODS.items():
                path, bandwidth = checkpoint_path(
                    directory, dataset, quantile, method_key
                )
                relative_path = path.resolve().relative_to(model_root).as_posix()
                observed_hash = sha256(path)
                expected_hash = expected_hashes.get(relative_path)
                if observed_hash != expected_hash:
                    raise RuntimeError(
                        f"Checkpoint SHA-256 mismatch for {relative_path}: "
                        f"expected {expected_hash}, found {observed_hash}"
                    )
                model = load_model(path)
                prediction = prediction_vector(model, x_test)
                summary, pointwise = result_frames(
                    dataset,
                    method_name,
                    quantile,
                    bandwidth,
                    y_test,
                    prediction,
                )
                rows.append(summary)
                pointwise_frames.append(pointwise)
                checkpoints.append(
                    {
                        "dataset": dataset,
                        "method": method_name,
                        "quantile": quantile,
                        "bandwidth": bandwidth,
                        "model_file": relative_path,
                        "sha256": observed_hash,
                    }
                )

    observed_files = {item["model_file"] for item in checkpoints}
    if observed_files != set(expected_hashes):
        missing = sorted(set(expected_hashes).difference(observed_files))
        extra = sorted(observed_files.difference(expected_hashes))
        raise RuntimeError(
            f"Checkpoint manifest mismatch; missing={missing}, extra={extra}"
        )

    output = pd.DataFrame(rows)
    pointwise = pd.concat(pointwise_frames, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.pointwise_output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    pointwise.to_csv(args.pointwise_output, index=False, compression="gzip")

    metadata_output = args.metadata_output or args.output.with_name(
        "saved_model_metadata.json"
    )
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.write_text(
        json.dumps(
            {
                "retrained": False,
                "split_seed": args.seed,
                "model_count": len(checkpoints),
                "prediction_rows": len(pointwise),
                "torch_version": torch.__version__,
                "manifest_file": str(manifest_path),
                "manifest_sha256": sha256(manifest_path),
                "checkpoints": checkpoints,
            },
            indent=2,
        )
        + "\n"
    )

    pointwise_inference, pooled_inference = write_table2_outputs(
        pointwise,
        args.inference_output_dir,
        chunk_size=args.bootstrap_chunk_size,
    )
    print(
        output.pivot(
            index=["dataset", "method"],
            columns="quantile",
            values="pinball_loss",
        )
    )
    print(pointwise_inference.to_string(index=False))
    print(pooled_inference.to_string(index=False))
    print(f"Saved {args.output}")
    print(f"Saved {args.pointwise_output}")
    print(f"Saved {metadata_output}")
    print(f"Saved Table 2 inference files under {args.inference_output_dir}")


if __name__ == "__main__":
    main()
