# Architecture

## Statistical experiment

The model defines a density on proxy-spin space and pulls that density back to
unconstrained latent coordinates. This order matters:

```text
z in R^N
  │  x_i = 1/2 + Softplus(z_i)
  ▼
proxy spins x in (1/2, infinity)^N
  ├── relaxed area A_tilde(x)
  ├── Gaussian area-shell penalty
  ├── periodic half-integer comb
  └── factorized degeneracy proxy
  │
  ├── Softplus Jacobian correction
  ▼
proper latent potential U(z)
  │
  ├── torch.autograd.grad
  ├── leapfrog integration
  ├── Metropolis correction
  └── warmup-only adaptation
  ▼
retained relaxed HMC draws
  │
  ├── area-shell diagnostics
  ├── lattice-distance diagnostics
  ├── permutation-invariant occupancies
  └── scalable curvature diagnostics
```

## Module ownership

| Module | Responsibility |
|---|---|
| `model.py` | Physical transformation, area operator, proper potential, analytic gradient |
| `hmc.py` | Metric, autograd evaluation, leapfrog, MH correction, warmup adaptation |
| `diagnostics.py` | ESS, area, lattice, occupancy, and curvature summaries |
| `artifacts.py` | Atomic JSON/tensor artifacts and provenance |
| `cli.py` | Configuration validation and reproducible experiment execution |

## Complexity

The potential and gradient require `O(N)` work and memory. In proxy-spin
coordinates the Hessian is diagonal plus rank one because the macroscopic area
constraint is the only all-to-all interaction in the baseline surrogate.

A dense metric requires `O(N^2)` memory and should not be used for thousands of
punctures. The preferred production metric is diagonal plus one low-rank area
correction, whose inverse can be applied with the Sherman–Morrison identity.

## Exactness boundary

The relaxed Markov chain is exact for its stated continuous target after the
Metropolis correction. It is not an exact sampler of the discrete LQG horizon
Hilbert space. That further claim requires:

1. discrete configuration recovery;
2. exact area-window enforcement;
3. invariant-intertwiner or finite-level Chern–Simons degeneracy evaluation;
4. correction for the relaxed proposal and rounding-cell probability.
