# Contributing

## Scientific standards

Every contribution must preserve the distinction among:

1. the exact discrete LQG horizon state space;
2. the continuous relaxed proposal distribution;
3. the factorized degeneracy proxy; and
4. any corrected intertwiner or Chern–Simons state count.

Do not label a posterior expectation as entropy or transfer a result to string
theory without defining the required normalizing constant, physical ensemble,
and exact correction.

## Development workflow

1. Create a feature branch.
2. Add or update unit tests.
3. Run `ruff check .`.
4. Run `python -m unittest discover -s tests -v`.
5. Run `python run_research_pipeline.py`.
6. Explain any changed physical assumption or estimand in the pull request.

Numerical changes should report the seed, configuration hash, PyTorch version,
acceptance rate, divergences, and effective sample sizes.
