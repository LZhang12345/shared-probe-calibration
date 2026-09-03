# Shared-probe calibration

This is a small reference implementation of the method in the MATS project
write-up. It contains the method itself, a CPU demonstration, and focused unit
tests. It deliberately leaves out comparison baselines, model downloads,
cluster launchers, and experiment outputs.

## Problem

Let $F:\mathbb{R}^N\rightarrow\mathbb{R}$ be a scalar model metric, where
$z_i$ scales an additive intervention on component $i$. At $z=0$, define the
attribution-patching vector $g$ and intervention-coordinate Hessian $G$ as

$$
g_i=\left.\frac{\partial F}{\partial z_i}\right|_{z=0},
\qquad
G_{ij}=\left.\frac{\partial^2F}{\partial z_i\,\partial z_j}\right|_{z=0}.
$$

The second-order estimate of the isolated intervention effect for component
$i$ is

$$
\widehat{\Delta}^{(2)}_i=g_i+\frac{1}{2}G_{ii}.
$$

Computing every $G_{ii}$ exactly costs one basis HVP per component. Instead,
draw a Rademacher vector $s\in\{-1,+1\}^N$. One shared HVP gives

$$
\widehat d(s)=s\odot(Gs),
\qquad
\mathbb{E}_s[\widehat d(s)]=\operatorname{diag}(G).
$$

With $R$ independent probes, the diagonal estimate is

$$
\widehat{\operatorname{diag}}(G)
=\frac{1}{R}\sum_{r=1}^{R}s^{(r)}\odot\bigl(Gs^{(r)}\bigr).
$$

One probe gives a noisy observation for every component. These observations
are not accurate enough to use as per-prompt corrections. The method therefore
uses them only to learn which components tend to have large diagonal curvature
across repeated analyses of the same model and task.

## Method

1. Run a small number of shared probes on each calibration instance.
2. For each component, predict $\log|G_{ii}|$ from $\log|g_i|$.
3. Partially pool each component fit toward a fit over all components. The
   implementation uses the fixed weight `n / (n + 5)`, matching the experiments.
4. On a new instance, rank components by predicted absolute curvature.
5. Spend the available basis HVPs on the selected components and apply their
   exact second-order corrections.

The randomized diagonal estimator is established prior art. The research
question is whether repeated intervention structure makes its noisy outputs
useful for amortized component selection.

## Install and run

Python 3.10 or newer is required. The example runs on CPU.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
pytest -q
python examples/toy_demo.py
```

The demonstration builds a small analytic intervention problem, gathers noisy
shared-probe observations, fits the selector, and applies exact HVP corrections
only to the selected coordinates.

## Main API

```python
from shared_probe_calibration import (
    CalibrationRecord,
    CurvatureCalibrator,
    InterventionSpace,
    correct_selected,
    exact_selected_diagonal,
    shared_probe_diagonal,
)

# metric_fn(z) must return a differentiable scalar. Interventions represented
# by z should be added to live activations rather than overwriting them.
space = InterventionSpace(metric_fn, n_components=N)
atp = space.gradient()
probe = shared_probe_diagonal(space, n_probes=8, seed=0)

record = CalibrationRecord(
    component_ids=component_ids,
    atp=atp.numpy(),
    probe_diagonal=probe.mean.numpy(),
)
calibrator = CurvatureCalibrator().fit(calibration_records)
selected = calibrator.select(atp.numpy(), component_ids, budget=B)

diagonal = exact_selected_diagonal(space, selected)
estimate = correct_selected(atp.numpy(), selected, diagonal.numpy())
```

In the reported experiments, the frozen recipe uses 50 calibration instances,
eight probes per instance, and a pooling strength of five. The classes remain
configurable so the implementation can be inspected on smaller examples.

## Scope

This package demonstrates the estimator and allocation rule. It does not
reproduce the reported model-scale numbers by itself; those runs require model
weights, task data, activation hooks, and substantially more compute. The
technical write-up reports the full experimental protocol, comparisons, and
negative results.
