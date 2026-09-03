import torch

from shared_probe_calibration import (
    InterventionSpace,
    exact_selected_diagonal,
    shared_probe_diagonal,
)


def quadratic_space(gradient: torch.Tensor, hessian: torch.Tensor) -> InterventionSpace:
    def metric(z: torch.Tensor) -> torch.Tensor:
        return gradient @ z + 0.5 * z @ hessian @ z

    return InterventionSpace(metric, gradient.numel(), dtype=torch.float64)


def test_hvp_and_selected_diagonal_match_analytic_quadratic() -> None:
    gradient = torch.tensor([0.4, -0.2, 0.8], dtype=torch.float64)
    hessian = torch.tensor(
        [[1.0, 0.2, -0.1], [0.2, 2.0, 0.3], [-0.1, 0.3, -0.5]],
        dtype=torch.float64,
    )
    space = quadratic_space(gradient, hessian)

    vector = torch.tensor([0.5, -1.0, 0.25], dtype=torch.float64)
    assert torch.allclose(space.gradient(), gradient)
    assert torch.allclose(space.hvp(vector), hessian @ vector)
    assert torch.allclose(
        exact_selected_diagonal(space, [2, 0]), hessian.diagonal()[[2, 0]]
    )


def test_true_effects_equal_second_order_estimate_for_quadratic() -> None:
    gradient = torch.tensor([0.4, -0.2, 0.8], dtype=torch.float64)
    hessian = torch.diag(torch.tensor([1.0, 2.0, -0.5], dtype=torch.float64))
    space = quadratic_space(gradient, hessian)
    expected = gradient + 0.5 * hessian.diagonal()
    assert torch.allclose(space.true_isolated_effects(), expected)


def test_shared_probe_estimate_is_exact_for_diagonal_hessian() -> None:
    gradient = torch.zeros(4, dtype=torch.float64)
    diagonal = torch.tensor([1.0, -2.0, 0.5, 4.0], dtype=torch.float64)
    space = quadratic_space(gradient, torch.diag(diagonal))
    result = shared_probe_diagonal(space, n_probes=3, seed=7)
    assert result.n_hvps == 3
    assert result.per_probe.shape == (3, 4)
    assert torch.allclose(result.mean, diagonal)


def test_selected_diagonal_rejects_bad_indices() -> None:
    space = quadratic_space(torch.zeros(2), torch.eye(2))
    try:
        exact_selected_diagonal(space, [0, 0])
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate indices should fail")

