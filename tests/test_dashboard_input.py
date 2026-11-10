import numpy as np
import pytest

from app.dashboard_input import FIELDS, MISSING, OUTSIDE, InputError, assessment_frame, preprocessing_steps
from src.data.load_data import load_config

NAMES = load_config()["feature_columns"]
SAMPLE = dict(zip(NAMES, [7.2, 4.8, 6.2, 320]))


def test_valid_frame_and_frozen_artifact():
    frame = assessment_frame(dict(reversed(SAMPLE.items())))
    assert frame.shape == (1, 4)
    assert frame.columns.tolist() == NAMES
    assert frame.dtypes.astype(str).tolist() == ["float64"] * 4
    transformed = preprocessing_steps().transform(frame)
    assert transformed.shape == (1, 5)
    assert np.isfinite(transformed).all()


def test_fields_and_inclusive_bounds():
    assert [field["name"] for field in FIELDS] == NAMES
    for key in ("min", "max"):
        frame = assessment_frame({f["name"]: f[key] for f in FIELDS})
        assert frame.shape == (1, 4)


@pytest.mark.parametrize("value", [None, ""])
def test_missing_input(value):
    with pytest.raises(InputError) as exc:
        assessment_frame({**SAMPLE, NAMES[1]: value})
    assert str(exc.value) == MISSING


@pytest.mark.parametrize("value", [20, -1, True, float("inf"), float("nan"), "7.2"])
def test_invalid_input(value):
    with pytest.raises(InputError) as exc:
        assessment_frame({**SAMPLE, NAMES[0]: value})
    assert str(exc.value) == OUTSIDE


def test_unexpected_field():
    with pytest.raises(InputError):
        assessment_frame({**SAMPLE, "wqi_score": 30})
