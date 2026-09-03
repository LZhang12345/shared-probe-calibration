"""Intervention-coordinate HVP utilities.

The caller defines a scalar function F(z). Coordinate z_i should scale an
additive intervention on component i. Additive live-activation semantics are
important: replacing a downstream activation with a fixed value could erase
the propagated effect of an upstream coordinate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch


@dataclass(frozen=True)
class ProbeEstimate:
    """A shared-probe estimate of the Hessian diagonal."""

    mean: torch.Tensor
    per_probe: torch.Tensor
    n_hvps: int


class InterventionSpace:
    """A differentiable scalar metric over intervention coordinates."""

    def __init__(
        self,
        metric_fn: Callable[[torch.Tensor], torch.Tensor],
        n_components: int,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        if n_components < 1:
            raise ValueError("n_components must be positive")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be torch.float32 or torch.float64")
        self.metric_fn = metric_fn
        self.n_components = int(n_components)
        self.device = torch.device(device)
        self.dtype = dtype

    def _point(self, *, requires_grad: bool = False) -> torch.Tensor:
        return torch.zeros(
            self.n_components,
            device=self.device,
            dtype=self.dtype,
            requires_grad=requires_grad,
        )

    def _graph_gradient(self) -> tuple[torch.Tensor, torch.Tensor]:
        z = self._point(requires_grad=True)
        output = self.metric_fn(z)
        if output.ndim != 0:
            raise ValueError("metric_fn must return a scalar tensor")
        (gradient,) = torch.autograd.grad(output, z, create_graph=True)
        return gradient, z

    def gradient(self) -> torch.Tensor:
        """Return the first derivative at z=0 (the AtP vector)."""

        gradient, _ = self._graph_gradient()
        return gradient.detach()

    def hvp(
        self,
        vector: torch.Tensor,
        *,
        graph_gradient: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Return G @ vector at z=0 using one Hessian-vector product."""

        vector = torch.as_tensor(vector, device=self.device, dtype=self.dtype)
        if vector.shape != (self.n_components,):
            raise ValueError(
                f"vector must have shape ({self.n_components},), got {tuple(vector.shape)}"
            )
        gradient, z = graph_gradient or self._graph_gradient()
        if not gradient.requires_grad:
            return torch.zeros_like(vector)
        (product,) = torch.autograd.grad(
            (gradient * vector).sum(), z, retain_graph=True
        )
        return product.detach()

    def true_isolated_effects(self) -> torch.Tensor:
        """Return F(e_i)-F(0) for every component using direct evaluations."""

        origin = self._point()
        base = self.metric_fn(origin).detach()
        effects = torch.empty(
            self.n_components, device=self.device, dtype=self.dtype
        )
        for index in range(self.n_components):
            point = self._point()
            point[index] = 1.0
            effects[index] = self.metric_fn(point).detach() - base
        return effects


def shared_probe_diagonal(
    space: InterventionSpace,
    n_probes: int,
    *,
    seed: int,
) -> ProbeEstimate:
    """Estimate diag(G) with shared Rademacher probes.

    For a probe s with independent entries in {-1, +1}, s * (G @ s) is an
    unbiased Hessian-diagonal estimate. Each probe costs one HVP in the joint
    intervention space, regardless of the number of components.
    """

    if n_probes < 1:
        raise ValueError("n_probes must be positive")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    graph_gradient = space._graph_gradient()
    estimates = torch.empty(
        n_probes,
        space.n_components,
        device=space.device,
        dtype=space.dtype,
    )
    for probe_index in range(n_probes):
        bits = torch.randint(
            0,
            2,
            (space.n_components,),
            generator=generator,
            dtype=torch.int8,
        )
        probe = (2.0 * bits.to(space.dtype) - 1.0).to(space.device)
        estimates[probe_index] = probe * space.hvp(
            probe, graph_gradient=graph_gradient
        )
    return ProbeEstimate(
        mean=estimates.mean(dim=0),
        per_probe=estimates,
        n_hvps=n_probes,
    )


def exact_selected_diagonal(
    space: InterventionSpace,
    selected_indices: Sequence[int],
) -> torch.Tensor:
    """Compute exact G_ii only for selected components.

    The cost is one basis HVP per selected component. Returned values follow
    the order of ``selected_indices``.
    """

    selected = [int(index) for index in selected_indices]
    if len(set(selected)) != len(selected):
        raise ValueError("selected_indices must not contain duplicates")
    if any(index < 0 or index >= space.n_components for index in selected):
        raise IndexError("selected component index is out of range")
    if not selected:
        return torch.empty(0, device=space.device, dtype=space.dtype)

    graph_gradient = space._graph_gradient()
    diagonal = torch.empty(len(selected), device=space.device, dtype=space.dtype)
    for output_index, component_index in enumerate(selected):
        basis = torch.zeros(
            space.n_components, device=space.device, dtype=space.dtype
        )
        basis[component_index] = 1.0
        diagonal[output_index] = space.hvp(
            basis, graph_gradient=graph_gradient
        )[component_index]
    return diagonal

