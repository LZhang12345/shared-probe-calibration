import numpy as np
import pytest

from shared_probe_calibration import (
    CalibrationRecord,
    CurvatureCalibrator,
    correct_selected,
)


def records() -> list[CalibrationRecord]:
    component_ids = ("a", "b", "c")
    output = []
    for scale in (0.5, 1.0, 2.0, 4.0):
        atp = np.array([scale, 2.0 * scale, 3.0 * scale])
        probe = np.array([2.0 * scale, 1.0 * scale, 4.0 * scale])
        output.append(CalibrationRecord(component_ids, atp, probe))
    return output


def test_calibrator_predicts_and_selects_deterministically() -> None:
    calibrator = CurvatureCalibrator(pooling_strength=5.0).fit(records())
    atp = np.array([1.0, 2.0, 3.0])
    prediction = calibrator.predict(atp, ("a", "b", "c"))
    assert prediction.shape == (3,)
    assert np.all(prediction > 0)
    first = calibrator.select(atp, ("a", "b", "c"), budget=2)
    second = calibrator.select(atp, ("a", "b", "c"), budget=2)
    assert first.tolist() == [2, 1]
    assert np.array_equal(first, second)


def test_unseen_component_uses_pooled_fit() -> None:
    calibrator = CurvatureCalibrator().fit(records())
    predicted = calibrator.predict(np.array([1.5]), ("unseen",))
    pooled = calibrator.pooled_coefficient_
    expected = np.exp(pooled[0] * np.log(1.5 + calibrator.epsilon) + pooled[1])
    assert np.allclose(predicted, [expected])


def test_correct_selected_changes_only_selected_entries() -> None:
    atp = np.array([1.0, 2.0, 3.0])
    corrected = correct_selected(atp, [2, 0], np.array([4.0, -2.0]))
    assert np.allclose(corrected, [0.0, 2.0, 5.0])
    assert np.allclose(atp, [1.0, 2.0, 3.0])


def test_fit_rejects_misaligned_record() -> None:
    bad = CalibrationRecord(("a",), np.array([1.0, 2.0]), np.array([1.0]))
    with pytest.raises(ValueError, match="align"):
        CurvatureCalibrator().fit([bad])


def test_prediction_requires_fit() -> None:
    with pytest.raises(RuntimeError, match="fit"):
        CurvatureCalibrator().predict(np.array([1.0]), ("a",))
