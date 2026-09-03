"""Cross-instance calibration and budgeted component selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class CalibrationRecord:
    """One calibration instance.

    ``probe_diagonal`` is the noisy shared-probe estimate, not an exact Hessian
    diagonal. ``component_ids`` must identify the same components across
    calibration and later instances.
    """

    component_ids: Sequence[Hashable]
    atp: np.ndarray
    probe_diagonal: np.ndarray


def _linear_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    design = np.column_stack((x, np.ones_like(x)))
    coefficient, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coefficient


class CurvatureCalibrator:
    """Predict absolute diagonal curvature from AtP and component identity.

    Component-specific log-linear fits are partially pooled toward one global
    fit. The fixed pooling weight is n/(n+pooling_strength), where n is the
    number of observations for that component.
    """

    def __init__(
        self,
        *,
        pooling_strength: float = 5.0,
        minimum_local_samples: int = 3,
        epsilon: float = 1e-12,
    ) -> None:
        if pooling_strength < 0:
            raise ValueError("pooling_strength must be nonnegative")
        if minimum_local_samples < 2:
            raise ValueError("minimum_local_samples must be at least two")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.pooling_strength = float(pooling_strength)
        self.minimum_local_samples = int(minimum_local_samples)
        self.epsilon = float(epsilon)
        self.pooled_coefficient_: np.ndarray | None = None
        self.component_coefficients_: dict[Hashable, np.ndarray] = {}

    def fit(self, records: Iterable[CalibrationRecord]) -> "CurvatureCalibrator":
        observations: dict[Hashable, list[tuple[float, float]]] = {}
        record_count = 0
        for record in records:
            ids = tuple(record.component_ids)
            atp = np.asarray(record.atp, dtype=np.float64)
            probe = np.asarray(record.probe_diagonal, dtype=np.float64)
            if atp.ndim != 1 or probe.ndim != 1:
                raise ValueError("atp and probe_diagonal must be one-dimensional")
            if len(ids) != atp.size or probe.size != atp.size:
                raise ValueError("component_ids, atp, and probe_diagonal must align")
            if not np.isfinite(atp).all() or not np.isfinite(probe).all():
                raise ValueError("calibration arrays must contain finite values")
            for component_id, atp_value, probe_value in zip(ids, atp, probe):
                observations.setdefault(component_id, []).append(
                    (abs(float(atp_value)), abs(float(probe_value)))
                )
            record_count += 1

        if record_count == 0 or not observations:
            raise ValueError("at least one nonempty calibration record is required")

        pooled_x = np.log(
            np.concatenate(
                [np.asarray([pair[0] for pair in values]) for values in observations.values()]
            )
            + self.epsilon
        )
        pooled_y = np.log(
            np.concatenate(
                [np.asarray([pair[1] for pair in values]) for values in observations.values()]
            )
            + self.epsilon
        )
        pooled = _linear_fit(pooled_x, pooled_y)

        component_coefficients: dict[Hashable, np.ndarray] = {}
        for component_id, values in observations.items():
            x = np.log(np.asarray([pair[0] for pair in values]) + self.epsilon)
            y = np.log(np.asarray([pair[1] for pair in values]) + self.epsilon)
            if len(values) >= self.minimum_local_samples and x.std() > 1e-8:
                local = _linear_fit(x, y)
            else:
                local = pooled
            weight = len(values) / (len(values) + self.pooling_strength)
            component_coefficients[component_id] = (
                weight * local + (1.0 - weight) * pooled
            )

        self.pooled_coefficient_ = pooled
        self.component_coefficients_ = component_coefficients
        return self

    def _require_fitted(self) -> np.ndarray:
        if self.pooled_coefficient_ is None:
            raise RuntimeError("fit must be called before prediction")
        return self.pooled_coefficient_

    def predict(
        self,
        atp: np.ndarray,
        component_ids: Sequence[Hashable],
    ) -> np.ndarray:
        """Predict nonnegative absolute diagonal curvature."""

        pooled = self._require_fitted()
        atp = np.asarray(atp, dtype=np.float64)
        ids = tuple(component_ids)
        if atp.ndim != 1 or len(ids) != atp.size:
            raise ValueError("component_ids and one-dimensional atp must align")
        if not np.isfinite(atp).all():
            raise ValueError("atp must contain finite values")
        coefficients = np.asarray(
            [self.component_coefficients_.get(component_id, pooled) for component_id in ids]
        )
        log_atp = np.log(np.abs(atp) + self.epsilon)
        return np.exp(coefficients[:, 0] * log_atp + coefficients[:, 1])

    def select(
        self,
        atp: np.ndarray,
        component_ids: Sequence[Hashable],
        *,
        budget: int,
    ) -> np.ndarray:
        """Return the deterministically ranked indices of the top predictions."""

        scores = self.predict(atp, component_ids)
        if budget < 0 or budget > scores.size:
            raise ValueError("budget must be between zero and the component count")
        order = np.argsort(-scores, kind="stable")
        return order[:budget]


def correct_selected(
    atp: np.ndarray,
    selected_indices: Sequence[int],
    selected_diagonal: np.ndarray,
) -> np.ndarray:
    """Apply exact second-order corrections to the selected AtP entries."""

    estimate = np.asarray(atp, dtype=np.float64).copy()
    if estimate.ndim != 1 or not np.isfinite(estimate).all():
        raise ValueError("atp must be a finite one-dimensional array")
    selected = np.asarray(selected_indices, dtype=np.int64)
    diagonal = np.asarray(selected_diagonal, dtype=np.float64)
    if selected.ndim != 1 or diagonal.ndim != 1 or selected.size != diagonal.size:
        raise ValueError("selected_indices and selected_diagonal must align")
    if np.unique(selected).size != selected.size:
        raise ValueError("selected_indices must not contain duplicates")
    if np.any(selected < 0) or np.any(selected >= estimate.size):
        raise IndexError("selected component index is out of range")
    if not np.isfinite(diagonal).all():
        raise ValueError("selected_diagonal must contain finite values")
    estimate[selected] += 0.5 * diagonal
    return estimate

