"""Prepare deterministic disposable inputs for legacy tests using default dataset paths.

Run only in a clean CI checkout. The five HTTP acceptance cases use their own
real observations and the tracked classifier, independent of this fixture.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.compute_wqi import assign_potability, assign_wqi_class, compute_wqi_score
from src.data.load_data import load_config
from src.models.train import perform_split


def main():
    config = load_config()
    path = Path(config["dataset_path"])
    generated_dirs = [Path("data/raw"), path.parent, Path("data/processed")]
    if any(list(folder.glob("*.csv*")) for folder in generated_dirs) or Path("models/preprocessing_pipeline_v1.pkl").exists():
        raise RuntimeError("Use a clean checkout to avoid replacing local data or preprocessing artifacts.")

    rng = np.random.default_rng(42)
    n = 512
    ph, oxygen, bod, coliform = config["feature_columns"]
    frame = pd.DataFrame({
        ph: rng.uniform(6.6, 8.1, n),
        oxygen: rng.uniform(4, 14, n),
        bod: rng.lognormal(1, 1, n),
        coliform: rng.lognormal(5, 2, n),
        "State": ["Fixture state"] * n,
        "Station": [f"Fixture station {i % 8}" for i in range(n)],
        "Latitude": 20.0,
        "Longitude": 75.0,
        "sampling_date": pd.date_range("2021-01-01", periods=n).strftime("%Y-%m-%d"),
    })
    frame["year"] = frame["sampling_date"].str[:4].astype(int)
    frame.loc[::23, coliform] = np.nan
    frame["wqi_score"] = frame.apply(compute_wqi_score, axis=1)
    frame["wqi_class"] = frame["wqi_score"].map(assign_wqi_class)
    frame["Potability"] = frame["wqi_score"].map(assign_potability)
    frame.to_csv(path, index=False, mode="x")
    perform_split(save_artifacts=True)


if __name__ == "__main__":
    main()
