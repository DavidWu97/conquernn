"""Load the two real-data sources used in the paper."""

from __future__ import annotations

import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_ROOT = Path(__file__).resolve().parent / "data"
DATASETS = ("bmi_male", "bmi_female", "california_housing")
BMI_FEATURES = (
    "age",
    "daily_steps",
    "sleep_hours",
    "water_intake_l",
    "calories_consumed",
)


def load_bmi(gender: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the BMI predictors and response after the paper's filters."""
    if gender not in {"Male", "Female"}:
        raise ValueError("gender must be 'Male' or 'Female'")
    path = DATA_ROOT / "bmi" / "health_lifestyle_dataset.csv"
    frame = pd.read_csv(path)
    frame = frame[(frame["family_history"] == 0) & (frame["gender"] == gender)]
    return (
        frame.loc[:, BMI_FEATURES].to_numpy(dtype=np.float64),
        frame["bmi"].to_numpy(dtype=np.float64),
    )


def load_california_housing() -> tuple[np.ndarray, np.ndarray]:
    """Read the original archive and apply scikit-learn's transformations."""
    path = DATA_ROOT / "housing" / "cal_housing.tgz"
    with tarfile.open(path, mode="r:gz") as archive:
        member = next(
            item
            for item in archive.getmembers()
            if Path(item.name).name == "cal_housing.data"
        )
        stream = archive.extractfile(member)
        if stream is None:
            raise OSError(f"Cannot read {member.name} from {path}")
        raw = np.loadtxt(stream, delimiter=",")

    if raw.shape != (20640, 9):
        raise ValueError(f"Unexpected California Housing shape: {raw.shape}")
    reordered = raw[:, [8, 7, 2, 3, 4, 5, 6, 1, 0]].copy()
    target = reordered[:, 0] / 100000.0
    features = reordered[:, 1:]
    households = features[:, 5].copy()
    features[:, 2] /= households
    features[:, 3] /= households
    features[:, 5] = features[:, 4] / households
    return features, target


def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray]:
    if name == "bmi_male":
        return load_bmi("Male")
    if name == "bmi_female":
        return load_bmi("Female")
    if name == "california_housing":
        return load_california_housing()
    raise ValueError(f"Unknown dataset {name!r}; choose from {DATASETS}")


def paper_split(
    name: str, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the fixed 80/20 split used in the camera-ready table."""
    features, target = load_dataset(name)
    return train_test_split(features, target, test_size=0.2, random_state=seed)
