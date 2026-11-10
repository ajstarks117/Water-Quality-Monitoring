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
    records["month"] = dates.dt.strftime("%Y-%m")
    stations = (frame.groupby(["State", "Station"])[["Latitude", "Longitude"]]
                .median().reset_index())
    return {
        "features": features,
        "stations": stations.astype(object).where(stations.notna(), None).to_dict("records"),
        "distribution": frame["wqi_class"].value_counts().to_dict(),
        "records": records.astype(object).where(records.notna(), None).to_dict("records"),
    }


def historical_dashboard():
    path = get_project_root() / load_config()["dataset_path"]
    return _read_history(path, path.stat().st_mtime_ns)
