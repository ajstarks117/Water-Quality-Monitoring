"""Assessment inputs use inclusive plausible bounds from Docs/data_dictionary.md."""
from functools import lru_cache
from math import isfinite

from src.models.predict import FROZEN_RAW_FEATURE_COLUMNS, load_model, validate_raw_features


BOUNDS = {
    "Potential of Hydrogen (pH)": (0, 14),
    "Dissolved oxygen (mg/L)": (0, 30),
    "Biochemical Oxygen Demand (mg/L)": (0, 200),
    "Fecal Coliform (MPN/100mL)": (0, 100_000_000),
}
FIELDS = [{"name": name, "min": BOUNDS[name][0], "max": BOUNDS[name][1]}
          for name in FROZEN_RAW_FEATURE_COLUMNS]
MISSING = "Please provide all required parameters."
OUTSIDE = "Input value is outside the allowed data range. Please verify the measurement."


class InputError(ValueError):
    pass


def assessment_frame(values):
    if not isinstance(values, dict) or any(values.get(k) in (None, "") for k in BOUNDS):
        raise InputError(MISSING)
    if values.keys() != BOUNDS.keys():
        raise InputError("Submit only the four required parameters.")
    for name, (low, high) in BOUNDS.items():
        value = values[name]
        if type(value) not in (int, float) or not low <= value <= high or not isfinite(value):
            raise InputError(OUTSIDE)
    return validate_raw_features(values).astype("float64")


@lru_cache(maxsize=1)
def frozen_model():
    return load_model()


def preprocessing_steps():
    return frozen_model()[:-1]


def preprocess_assessment(values):
    frame = assessment_frame(values)
    transformed = preprocessing_steps().transform(frame)
    return {"input_count": frame.shape[1], "processed_feature_count": transformed.shape[1]}
