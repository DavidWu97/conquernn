"""Compute the paired-bootstrap statistics reported with camera-ready Table 2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
BASELINE = "Nonsmooth QRNN"
CONQUER_METHODS = (
    "ConquerNet--Gaussian",
    "ConquerNet--Uniform",
    "ConquerNet--Epanechnikov",
)
DATASETS = ("bmi_male", "bmi_female", "california_housing")
REQUIRED_COLUMNS = (
    "dataset",
    "test_row",
    "quantile",
    "target",
    "method",
    "prediction",
    "pinball_loss",
)
POINTWISE_PROTOCOLS = {
    "bmi_male": (20_000, 20260718),
    "bmi_female": (20_000, 20260719),
    "california_housing": (100_000, 20260719),
}
POOLED_PROTOCOLS = {
    "BMI": {
        "datasets": ("bmi_male", "bmi_female"),
        "repeats": 100_000,
        "seed": 20260721,
        "p_value": "centered_absolute_two_sided",
    },
    "California Housing": {
        "datasets": ("california_housing",),
        "repeats": 100_000,
        "seed": 20260720,
        "p_value": "equal_tail_inverting_basic_ci",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Long-form per-test-point predictions written by real_data_benchmark.py.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--bootstrap-chunk-size", type=int, default=100)
    args = parser.parse_args()
    if not 0.0 < args.confidence_level < 1.0:
        parser.error("--confidence-level must be between 0 and 1")
    if args.bootstrap_chunk_size < 1:
        parser.error("--bootstrap-chunk-size must be positive")
    return args


def pointwise_pinball(
    target: np.ndarray, prediction: np.ndarray, quantile: float
) -> np.ndarray:
    target = np.asarray(target)
    prediction = np.asarray(prediction)
    residual = target - prediction
    return np.maximum(quantile * residual, (quantile - 1.0) * residual)


def validate_pointwise(frame: pd.DataFrame) -> None:
    missing = sorted(set(REQUIRED_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"Pointwise results are missing columns: {missing}")
    if frame.loc[:, REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("Pointwise results contain missing required values")
    if not np.isfinite(
        frame[["quantile", "target", "prediction", "pinball_loss"]].to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError("Pointwise results contain non-finite values")
    unknown = sorted(set(frame["dataset"]).difference(DATASETS))
    if unknown:
        raise ValueError(f"Unknown datasets in pointwise results: {unknown}")
    duplicated = frame.duplicated(["dataset", "test_row", "quantile", "method"])
    if duplicated.any():
        raise ValueError(
            "Pointwise results contain duplicate dataset-row-method entries"
        )

    expected_methods = {BASELINE, *CONQUER_METHODS}
    expected_quantiles = set(QUANTILES)
    for dataset, part in frame.groupby("dataset", sort=False):
        methods = set(part["method"])
        quantiles = set(part["quantile"].astype(float))
        if methods != expected_methods:
            raise ValueError(f"{dataset} has methods {sorted(methods)}")
        if quantiles != expected_quantiles:
            raise ValueError(f"{dataset} has quantiles {sorted(quantiles)}")
        counts = part.groupby("test_row", sort=False).size()
        expected = len(QUANTILES) * len(expected_methods)
        if not counts.eq(expected).all():
            raise ValueError(f"{dataset} has an incomplete subject-comparison matrix")


def _difference_matrix(
    frame: pd.DataFrame, dataset: str
) -> tuple[np.ndarray, list[tuple[float, str, float, float]]]:
    part = frame.loc[frame["dataset"] == dataset]
    difference_columns = []
    metadata = []
    for quantile in QUANTILES:
        quantile_frame = part.loc[
            np.isclose(part["quantile"].to_numpy(dtype=float), quantile)
        ]
        pivot = quantile_frame.pivot(
            index="test_row", columns="method", values="pinball_loss"
        ).sort_index()
        baseline = pivot[BASELINE].to_numpy(dtype=np.float64)
        for method in CONQUER_METHODS:
            comparison = pivot[method].to_numpy(dtype=np.float64)
            difference_columns.append(baseline - comparison)
            metadata.append(
                (quantile, method, float(baseline.mean()), float(comparison.mean()))
            )
    return np.column_stack(difference_columns), metadata


def _bootstrap_mean_matrix(
    values: np.ndarray, repeats: int, seed: int, chunk_size: int
) -> np.ndarray:
    n_subjects, n_comparisons = values.shape
    output = np.empty((repeats, n_comparisons), dtype=np.float64)
    probabilities = np.full(n_subjects, 1.0 / n_subjects, dtype=np.float64)
    rng = np.random.default_rng(seed)
    for start in range(0, repeats, chunk_size):
        stop = min(start + chunk_size, repeats)
        counts = rng.multinomial(n_subjects, probabilities, size=stop - start).astype(
            np.float64, copy=False
        )
        output[start:stop] = counts @ values / n_subjects
    return output


def _bootstrap_mean_vector(
    values: np.ndarray, repeats: int, seed: int, chunk_size: int
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    output = np.empty(repeats, dtype=np.float64)
    rng = np.random.default_rng(seed)
    for start in range(0, repeats, chunk_size):
        stop = min(start + chunk_size, repeats)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        output[start:stop] = values[indices].mean(axis=1)
    return output


def _basic_ci(
    bootstrap: np.ndarray, observed: np.ndarray | float, confidence_level: float
) -> tuple[np.ndarray, np.ndarray]:
    alpha = 1.0 - confidence_level
    lower_quantile, upper_quantile = np.quantile(
        bootstrap, [alpha / 2.0, 1.0 - alpha / 2.0], axis=0
    )
    return 2.0 * observed - upper_quantile, 2.0 * observed - lower_quantile


def _centered_p(bootstrap: np.ndarray, observed: np.ndarray | float) -> np.ndarray:
    centered = bootstrap - observed
    exceedances = np.sum(np.abs(centered) >= np.abs(observed), axis=0)
    return (exceedances + 1.0) / (len(bootstrap) + 1.0)


def _equal_tail_p(bootstrap: np.ndarray, observed: np.ndarray | float) -> np.ndarray:
    centered = bootstrap - observed
    left = (np.sum(centered <= observed, axis=0) + 1.0) / (len(bootstrap) + 1.0)
    right = (np.sum(centered >= observed, axis=0) + 1.0) / (len(bootstrap) + 1.0)
    return np.minimum(1.0, 2.0 * np.minimum(left, right))


def pointwise_inference(
    frame: pd.DataFrame, confidence_level: float = 0.95, chunk_size: int = 100
) -> pd.DataFrame:
    rows = []
    available = set(frame["dataset"])
    for dataset in DATASETS:
        if dataset not in available:
            continue
        differences, metadata = _difference_matrix(frame, dataset)
        repeats, seed = POINTWISE_PROTOCOLS[dataset]
        observed = differences.mean(axis=0)
        bootstrap = _bootstrap_mean_matrix(differences, repeats, seed, chunk_size)
        ci_lower, ci_upper = _basic_ci(bootstrap, observed, confidence_level)
        centered_p = _centered_p(bootstrap, observed)
        equal_tail_p = _equal_tail_p(bootstrap, observed)
        for index, (quantile, method, baseline_loss, method_loss) in enumerate(
            metadata
        ):
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "quantile": quantile,
                    "n_test": len(differences),
                    "baseline_loss": baseline_loss,
                    "method_loss": method_loss,
                    "improvement": observed[index],
                    "basic_ci_lower": ci_lower[index],
                    "basic_ci_upper": ci_upper[index],
                    "bootstrap_se": bootstrap[:, index].std(ddof=1),
                    "probability_improvement": np.mean(bootstrap[:, index] > 0.0),
                    "p_centered_two_sided": centered_p[index],
                    "p_equal_tail_two_sided": equal_tail_p[index],
                    "bootstrap_repeats": repeats,
                    "bootstrap_seed": seed,
                    "confidence_level": confidence_level,
                }
            )
    return pd.DataFrame(rows)


def pooled_inference(
    frame: pd.DataFrame, confidence_level: float = 0.95, chunk_size: int = 100
) -> pd.DataFrame:
    available = set(frame["dataset"])
    matrices = {
        dataset: _difference_matrix(frame, dataset)[0]
        for dataset in DATASETS
        if dataset in available
    }
    rows = []
    for estimand, protocol in POOLED_PROTOCOLS.items():
        if not set(protocol["datasets"]).issubset(matrices):
            continue
        subject_average = np.concatenate(
            [matrices[dataset].mean(axis=1) for dataset in protocol["datasets"]]
        )
        observed = float(subject_average.mean())
        bootstrap = _bootstrap_mean_vector(
            subject_average,
            protocol["repeats"],
            protocol["seed"],
            chunk_size,
        )
        ci_lower, ci_upper = _basic_ci(bootstrap, observed, confidence_level)
        centered_p = float(_centered_p(bootstrap, observed))
        equal_tail_p = float(_equal_tail_p(bootstrap, observed))
        selected_p = (
            centered_p
            if protocol["p_value"] == "centered_absolute_two_sided"
            else equal_tail_p
        )
        rows.append(
            {
                "estimand": estimand,
                "datasets": ",".join(protocol["datasets"]),
                "n_subjects": len(subject_average),
                "comparisons_per_subject": len(QUANTILES) * len(CONQUER_METHODS),
                "pooled_absolute_improvement": observed,
                "basic_ci_lower": float(ci_lower),
                "basic_ci_upper": float(ci_upper),
                "bootstrap_se": bootstrap.std(ddof=1),
                "probability_improvement": np.mean(bootstrap > 0.0),
                "p_value": selected_p,
                "p_value_definition": protocol["p_value"],
                "p_centered_two_sided": centered_p,
                "p_equal_tail_two_sided": equal_tail_p,
                "bootstrap_repeats": protocol["repeats"],
                "bootstrap_seed": protocol["seed"],
                "confidence_level": confidence_level,
            }
        )
    return pd.DataFrame(rows)


def write_table2_outputs(
    frame: pd.DataFrame,
    output_dir: Path,
    confidence_level: float = 0.95,
    chunk_size: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_pointwise(frame)
    pointwise = pointwise_inference(frame, confidence_level, chunk_size)
    pooled = pooled_inference(frame, confidence_level, chunk_size)
    output_dir.mkdir(parents=True, exist_ok=True)
    pointwise.to_csv(output_dir / "table2_pointwise_inference.csv", index=False)
    pooled.to_csv(output_dir / "table2_pooled_inference.csv", index=False)
    metadata = {
        "difference": "Nonsmooth QRNN pinball loss - ConquerNet pinball loss",
        "bootstrap_unit": "held-out test subject",
        "confidence_interval": "pointwise basic paired bootstrap",
        "confidence_level": confidence_level,
        "loss_precision": {
            "bmi_male": "float32",
            "bmi_female": "float32",
            "california_housing": "float64",
        },
        "pointwise_protocols": POINTWISE_PROTOCOLS,
        "pooled_protocols": POOLED_PROTOCOLS,
    }
    (output_dir / "table2_inference_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    return pointwise, pooled


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input)
    pointwise, pooled = write_table2_outputs(
        frame,
        args.output_dir,
        confidence_level=args.confidence_level,
        chunk_size=args.bootstrap_chunk_size,
    )
    print(pointwise.to_string(index=False))
    print(pooled.to_string(index=False))
    print(f"Saved Table 2 inference files under {args.output_dir}")


if __name__ == "__main__":
    main()
