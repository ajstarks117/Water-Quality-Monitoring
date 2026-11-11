"""Historical observations from the configured Phase 1 labeled CPCB dataset."""
from functools import lru_cache

import pandas as pd

from src.data.load_data import get_project_root, load_config, load_raw_data


@lru_cache(maxsize=1)
def _read_history(path, modified):
    frame = load_raw_data(path)
    features = load_config()["feature_columns"]
    dates = pd.to_datetime(frame["sampling_date"], format="%Y-%m-%d", errors="coerce")
    records = frame[["State", "Station", "wqi_class", *features]].copy()
    records.columns = ["state", "station", "category", *features]
    records["date"] = dates.dt.strftime("%Y-%m-%d")
    records["month"] = dates.dt.strftime("%Y-%m")
    keys = ["State", "Station"]
    # Keep a coordinate pair from one real observation, including when a station moved.
    located = frame["Latitude"].between(-90, 90) & frame["Longitude"].between(-180, 180)
    coordinates = (frame.loc[located, keys + ["Latitude", "Longitude"]]
                   .assign(sampled=dates[located]).sort_values("sampled", na_position="first")
                   .drop_duplicates(keys, keep="last"))
    stations = (frame[keys].drop_duplicates().sort_values(keys)
                .merge(coordinates.drop(columns="sampled"), on=keys, how="left"))
    return {
        "features": features,
        "stations": stations.astype(object).where(stations.notna(), None).to_dict("records"),
        "distribution": frame["wqi_class"].value_counts().to_dict(),
        "records": records.astype(object).where(records.notna(), None).to_dict("records"),
    }


def historical_dashboard():
    path = get_project_root() / load_config()["dataset_path"]
    return _read_history(path, path.stat().st_mtime_ns)
