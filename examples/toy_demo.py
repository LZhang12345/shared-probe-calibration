"""CPU demonstration of shared-probe calibration and selective correction."""

from __future__ import annotations

import numpy as np
import torch

from shared_probe_calibration import (
    CalibrationRecord,
    CurvatureCalibrator,
    InterventionSpace,
    correct_selected,
    exact_selected_diagonal,
    shared_probe_diagonal,
)


def quadratic_space(gradient: np.ndarray, hessian: np.ndarray) -> InterventionSpace:
    gradient_tensor = torch.as_tensor(gradient, dtype=torch.float64)
    hessian_tensor = torch.as_tensor(hessian, dtype=torch.float64)

    def metric(z: torch.Tensor) -> torch.Tensor:
        return gradient_tensor @ z + 0.5 * z @ hessian_tensor @ z

    return InterventionSpace(metric, len(gradient), dtype=torch.float64)


def make_instance(seed: int, n_components: int = 12) -> tuple[InterventionSpace, tuple[str, ...]]:
    rng = np.random.default_rng(seed)
    component_ids = tuple(f"head_{index}" for index in range(n_components))
    component_scale = np.linspace(0.3, 1.8, n_components)
    gradient = rng.lognormal(mean=-0.5, sigma=0.7, size=n_components)
    gradient *= rng.choice((-1.0, 1.0), size=n_components)
    diagonal = component_scale * np.sqrt(np.abs(gradient) + 0.05)

    noise = rng.normal(scale=0.08, size=(n_components, n_components))
    hessian = 0.5 * (noise + noise.T)
    np.fill_diagonal(hessian, diagonal)
    return quadratic_space(gradient, hessian), component_ids


def main() -> None:
    records = []
    for seed in range(50):
        space, component_ids = make_instance(seed)
        probe = shared_probe_diagonal(space, n_probes=8, seed=10_000 + seed)
        records.append(
            CalibrationRecord(
                component_ids=component_ids,
                atp=space.gradient().numpy(),
                probe_diagonal=probe.mean.numpy(),
            )
        )

    calibrator = CurvatureCalibrator(pooling_strength=5.0).fit(records)
    test_space, component_ids = make_instance(1_000)
    atp = test_space.gradient().numpy()
    selected = calibrator.select(atp, component_ids, budget=4)
    diagonal = exact_selected_diagonal(test_space, selected).numpy()
    estimate = correct_selected(atp, selected, diagonal)
    truth = test_space.true_isolated_effects().numpy()

    print("Selected components:", ", ".join(component_ids[i] for i in selected))
    print(f"AtP mean absolute error:       {np.abs(atp - truth).mean():.6f}")
    print(f"Selective-HVP absolute error:  {np.abs(estimate - truth).mean():.6f}")
    print("Calibration HVPs: 50 instances x 8 probes")
    print("Inference HVPs: 4")


if __name__ == "__main__":
    main()

