# Scientific and reproducibility audit

Audit initiated 2026-10-11 (America/New_York). Baseline commit:
`2ec91134699e00be73f49dbc80a5fe75bf32b9e5`.
Baseline tree: `05afb8246fb243bb8a622a97cb58dbb474968ffb`.

## Scientific scope

The fixed-puncture LQG-inspired **continuous** target is appropriate for the
stated research scope. The area operator, half-integer comb, factorized
degeneracy proxy, transformation Jacobian, leapfrog involution, and
Metropolis acceptance formula have the correct mathematical form.
The Gaussian area penalty is a soft constraint; neither relaxed nor rounded
draws enforce an exact microcanonical area window.

For fixed N and positive area stiffness, the area penalty confines the positive
proxy-spin tail and dominates the polynomial degeneracy factor. The negative
latent tail is confined by `-log sigmoid(z)`. The target is normalizable, but
the code does not estimate its normalizing constant or a physical state count.
It does not impose an SU(2) invariant-space or finite-level Chern-Simons count.

HMC is implemented. NUTS, replica exchange, multichain rank-normalized R-hat,
bulk/tail ESS, reference-normalizer estimation, and exact discrete correction
remain future work. This audit does not add or claim those methods.

## Confirmed defects and corrections

| Defect | Reproduction / consequence | Corrected behavior |
|---|---|---|
| Constant-trace ESS | 80 identical draws returned ESS = 80 | ESS is unidentified (`null`); stuck coordinates are reported and a fully rejected/stuck chain fails |
| Autocorrelation truncation | Pairing started at lags 1+2 and omitted the monotone restriction | Pair lags 0+1, 2+3, ... using the initial positive, monotone sequence; conservative cap at draws; FFT evaluation |
| Unit-dependent ESS | Tiny but varying traces were treated as constant by an absolute machine-epsilon cutoff | Normalize trace scale before autocorrelation; constant and <4-draw traces are explicitly unidentified |
| Thermodynamic sign | A test trace saved +0.5178176 as the log-normalizer integrand | Save `d log Z / d kappa_Q = -E[Q]`; the mean comb statistic remains positive |
| Unconditional sampling PASS | Failed/rejected/divergent or inadequate chains could be called PASS | Sampling states are FAIL, WARN, or DIAGNOSTICS_ONLY; convergence is NOT_ASSESSED_SINGLE_CHAIN |
| Incomplete numerical validation | NaN area targets, NaN step sizes, negative divergence thresholds and numeric booleans were accepted | Reject nonfinite controls, invalid integer/boolean controls, invalid vectors, and metric broadcasting before sampling |
| Thresholded transform derivative | Softplus's linear threshold was not the exact sigmoid derivative used by the analytic gradient | Use stable `logaddexp(z, 0)` and test tail gradients |
| Unbounded adaptation arithmetic | Extreme dual-averaging log step sizes could overflow/underflow | Bound log step during warmup; retained sampling still uses a frozen adapted kernel |
| Nonfinite artifact serialization | Tensor conversion could bypass scalar sanitization and emit NaN/Infinity; some scalar failures became null | Recursively reject nonfinite JSON; unidentified diagnostics use explicit null; clean temporary files on write failure |
| Unidentified sample files | A bare tensor could be analyzed under another same-dimensional configuration | Schema-2 envelope verifies sample-content, metadata, input-config and producer-source SHA-256 |
| Incorrect overridden-seed provenance | Parts 2/3 copied the configured seed rather than the actual sampling override | Preserve the producer's effective seed, unique run ID, environment and sample-file digest through Parts 1–3 |
| Path-dependent pipeline | Relative config paths were interpreted after changing to the repository directory | Resolve caller paths and support an isolated output root; run Part 4 from the repository root |
| Mixed coordinate curvature descriptions | Latent area-gradient curvature and proxy-spin comb curvature shared a summary | Explicitly identify coordinate conventions and partial-curvature proxies |
| Missing rounded-area diagnostics | Lattice rounding could be confused with exact area-window enforcement | Report rounded-area mean, SD and RMSE separately from relaxed area |
| Emulator scope/domain | Site named HMC / NUTS and its illustrative wobble could go below spin 1/2 | Identify implemented HMC versus planned NUTS; clamp the explanatory emulator's proxy domain |

Hashes establish reproducibility and detect accidental corruption; they are not
cryptographic attestations of who generated an artifact. The source digest
covers the installed package's Python modules. Schema-1 sample files cannot
receive new provenance by reanalysis: regenerate them using the full pipeline.

## Verification

The original 11 tests pass, and the expanded suite contains **50 tests**.
Independent numerical checks include:

- One-dimensional continuous target moments from direct quadrature versus HMC
  sample moments, with serial-dependence-aware Monte Carlo tolerance.
- Finite-difference log-normalizer derivative versus `-E[Q]`.
- IID versus AR(1) scalar ESS and invariance to measurement units.
- Leapfrog involution with unequal diagonal masses and autograd/analytic
  gradient agreement in both Softplus tails.
- Rejection of invalid configurations, nonfinite proposals, changed sample
  values, changed seed metadata, swapped configurations, and changed source.
- Repeated seeded sampling without modifying the caller's default dtype.
- Real CLI/pipeline subprocesses from a different working directory, with an
  overridden seed and disposable artifacts.

Local verification uses Python 3.12.14, PyTorch 2.14.1+cpu, the complete
`requirements-ci-lock.txt`, single-thread numerical libraries, full Ruff lint,
compilation, Parts 0–4, package build, YAML parsing, and `pip check`.
GitHub CI exercises the same lock and pipeline on Python 3.11/3.12 under Linux
and Windows. Each job runs the complete regression suite through Part 4.

## Current N = 64 exploratory baseline

The configuration and statistical target parameters are unchanged. The
corrected implementation in the locked local CPU environment yields:

| Quantity | Value |
|---|---:|
| Retained draws | 80 |
| Acceptance rate | 0.625 |
| Retained divergences / nonfinite energy errors | 0 / 0 |
| Relaxed area mean / target | 522.34709949 / 512 |
| Relaxed area RMSE | 10.96043960 |
| Area scalar ESS | 22.77779696 |
| Lattice RMS mean | 0.08487400 |
| Lattice scalar ESS | 10.14715573 |
| Potential scalar ESS | 4.59632371 |
| Rounded area mean / RMSE | 514.80934767 / 5.42087457 |
| d log Z / d kappa_Q point estimate | -13.39365749 |
| Sampling / convergence status | WARN / NOT_ASSESSED_SINGLE_CHAIN |

The <50 scalar ESS warning is an exploratory heuristic, not a validated
inferential threshold. Numerical stability, test success, and zero divergences
do not establish global equilibrium. In particular, the comb's coordinated
spin-sector barriers still require dispersed chains and tempering studies.
Monte Carlo standard errors are conditional scalar diagnostics and do not
account for unvisited modes. CPU/platform arithmetic can change adapted
trajectories; seeded reproducibility is tested within the same environment.

The existing APA PDF is preserved as a historical report of an earlier
environment and ESS estimator. Its .925 acceptance and 36.22/18.34 ESS values
are not re-certified by this audit and must not be used as current results.
The corrected baseline also remains exploratory. No black-hole entropy,
Bekenstein-Hawking coefficient, or exact microstate count is established here.

## Primary methodological references

- Ashtekar, Baez & Krasnov (2000), *Quantum Geometry of Isolated Horizons and
  Black Hole Entropy*: https://arxiv.org/abs/gr-qc/0005126
- Engle et al. (2010), SU(2)-invariant isolated-horizon formulation:
  https://arxiv.org/abs/1006.0634
- Hoffman & Gelman, *The No-U-Turn Sampler*:
  https://arxiv.org/abs/1111.4246
- Stan Reference Manual, posterior analysis (ESS, MCSE, multichain convergence):
  https://mc-stan.org/docs/reference-manual/analysis.html

The source code and independent numerical regressions support the reported
implementation corrections. These references support the scientific scope and
diagnostic interpretation, not a claim that this surrogate counts physical
black-hole microstates.
