"""Convert a simulation NPZ file to a tidy table."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from utils import get_idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def scalar(saved, key: str):
    return np.asarray(saved[key]).item()


def summarize(path: Path) -> pd.DataFrame:
    with np.load(path, allow_pickle=False) as saved:
        scenario = int(scalar(saved, "scenario"))
        shape = tuple(np.asarray(saved["shape"], dtype=int))
        methods = np.asarray(saved["methods"]).astype(str)
        sample_sizes = np.asarray(saved["sample_sizes"], dtype=int)
        quantiles = np.asarray(saved["quantiles"], dtype=float)
        bandwidths = np.asarray(saved["bandwidths"], dtype=float)
        metrics = {
            name: np.asarray(saved[name]) for name in ("mse", "mae", "coverage_bias")
        }

    if sample_sizes.tolist() != [1000, 5000, 10000]:
        raise ValueError(
            "The paper bandwidth map requires sample sizes 1000, 5000, 10000"
        )
    shape_index = 0 if shape == (5, 70) else 1 if shape == (10, 50) else None
    if shape_index is None:
        raise ValueError(f"The paper bandwidth map does not define shape {shape}")

    rows = []
    for sample_index, sample_size in enumerate(sample_sizes):
        bandwidth_index = get_idx(scenario - 1, shape_index, sample_index)
        for method_index, method in enumerate(methods):
            selected = 0 if method == "baseline" else bandwidth_index
            for quantile_index, quantile in enumerate(quantiles):
                row = {
                    "scenario": scenario,
                    "shape": f"{shape[0]}x{shape[1]}",
                    "sample_size": sample_size,
                    "quantile": quantile,
                    "method": method,
                    "bandwidth": (
                        np.nan if method == "baseline" else bandwidths[selected]
                    ),
                }
                index = (
                    slice(None),
                    selected,
                    method_index,
                    sample_index,
                    quantile_index,
                )
                for metric, values in metrics.items():
                    row[f"{metric}_mean"] = np.nanmean(values[index])
                    row[f"{metric}_sd"] = np.nanstd(values[index], ddof=1)
                rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    table = summarize(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
