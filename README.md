# Quantum Radiation Flow Decoder

[![CI](https://github.com/salemmorelli1/Quantum-Radiation-Flow-Decoder/actions/workflows/ci.yml/badge.svg)](https://github.com/salemmorelli1/Quantum-Radiation-Flow-Decoder/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-float64-EE4C2C.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Project 3 is a publication-oriented PyTorch research environment for testing a
specific computational question: how much of a toy information-bearing state can
be recovered from time-conditioned, strongly scrambled radiation telemetry by an
invertible deep density model?

The repository combines a finite-dimensional Page-curve benchmark, a frozen
conditional scrambling channel, a conditional RealNVP model, maximum-likelihood
and variational objectives, exact sample-wise log-density accounting, Monte Carlo
entropy estimation, and paired post-Page state-recovery diagnostics.

## Scientific boundary

This project distinguishes three quantities that are often conflated:

1. **Quantum entropy:** `S(rho_R) = -Tr(rho_R log rho_R)` for a radiation density
   operator or the finite-dimensional Page benchmark.
2. **Classical telemetry entropy:** `h(q_theta) = -E_q[log q_theta(X)]` for a
   continuous measurement distribution. The flow computes each `log q_theta(x)`
   exactly; its expectation is estimated by Monte Carlo.
3. **Toy decoder fidelity:** squared overlap between a paired synthetic secret
   state and a state reconstructed from inverse flow coordinates.

A classical normalizing flow over measurement outcomes does **not**, by itself,
represent a complex density operator, recover unmeasured phase information, or
prove unitary black-hole evaporation. Invertibility of the network is not the
same as invertibility of an unknown quantum channel. These distinctions are
enforced in the API, report, tests, and browser laboratory.

## Model

For normalized time `tau` and stopped mass `M(tau)`, the learned bijection is

```text
x = f_theta(z; tau, M),       z ~ Normal(0, I),
log q_theta(x | tau, M)
  = log phi(z) - log |det D_z f_theta(z; tau, M)|.
```

The reference telemetry channel uses a paired latent

```text
y_tau = v(tau) s + sqrt(1 - v(tau)^2) epsilon,
v(tau) = sigmoid((tau - tau_Page) / width),
```

where `s` is an information-bearing Gaussian coordinate vector and `epsilon` is
independent nuisance. The marginal distribution of `y_tau` remains standard
Gaussian, so the frozen reference scrambler has a normalized exact density;
correlation with the secret increases across Page time.

## Installation

```bash
git clone https://github.com/salemmorelli1/Quantum-Radiation-Flow-Decoder.git
cd Quantum-Radiation-Flow-Decoder
python -m venv .venv
source .venv/Scripts/activate  # Git Bash on Windows
python -m pip install --upgrade pip
python -m pip install -e ".[dev,report]"
```

For an environment that matches the GitHub verification matrix, install the
audited CI lock first and then install this package without re-resolving its
dependencies:

```bash
python -m pip install --disable-pip-version-check -r requirements-ci-lock.txt
python -m pip install --disable-pip-version-check --no-deps -e .
python -m pip check
```

## Run the research pipeline

```bash
python run_research_pipeline.py --config configs/baseline.json --output artifacts/latest
```

Quick smoke run:

```bash
python run_research_pipeline.py --steps 10 --output artifacts/smoke
```

The run writes:

- `experiment_summary.json`: configuration, optimization history, diagnostics,
  and interpretive labels.
- `page_curve.csv`: quantum benchmark, flow entropy, Monte Carlo standard error,
  KL estimates, visibility, mass, and decoder fidelity at each time.

## Tests and static checks

```bash
python -m pytest
python -m ruff check .
python -m compileall src tests scripts report
```

The tests verify exact forward/inverse round trips, Jacobian cancellation,
finite log densities, density-operator entropy, Page-curve shape, target-density
consistency, and a short differentiable training pass.

## Repository structure

```text
.
|-- configs/baseline.json
|-- docs/
|   |-- index.html
|   |-- Quantum_Radiation_Flow_APA_Report.pdf
|   |-- architecture.md
|   `-- research_blueprint.md
|-- scripts/
|   |-- part0_validate_model.py
|   |-- part1_train_conditional_flow.py
|   |-- part2_generate_page_curve.py
|   |-- part3_decrypt_state.py
|   `-- part4_validate_repository.py
|-- report/
|   |-- Quantum_Radiation_Flow_APA_Report.docx
|   |-- reproduce_report_figures.py
|   `-- build_report.py
|-- src/quantum_radiation_flow/
|   |-- config.py
|   |-- flow.py
|   |-- quantum.py
|   |-- target.py
|   |-- training.py
|   |-- diagnostics.py
|   `-- pipeline.py
|-- tests/
|-- run_research_pipeline.py
`-- pyproject.toml
```

## Interpretation protocol

A confirmatory run should preregister the seed family, flow depth, conditioning
variables, optimization budget, target measurement design, and pass/fail rules.
At minimum, report held-out forward and reverse KL estimates, entropy Monte Carlo
errors, round-trip error, decoder fidelity with confidence intervals, and
monotonicity violations on each side of the Page benchmark. Compare against
non-invertible and unconditioned baselines.

## Primary references

- Almheiri, A., Engelhardt, N., Marolf, D., & Maxfield, H. (2019). The entropy
  of bulk quantum fields and the entanglement wedge of an evaporating black
  hole. *Journal of High Energy Physics, 2019*, 63.
- Dinh, L., Sohl-Dickstein, J., & Bengio, S. (2017). Density estimation using
  Real NVP. *International Conference on Learning Representations*.
- Hayden, P., & Preskill, J. (2007). Black holes as mirrors: Quantum information
  in random subsystems. *Journal of High Energy Physics, 2007*(09), 120.
- Page, D. N. (1993). Average entropy of a subsystem. *Physical Review Letters,
  71*(9), 1291-1294.
- Papamakarios, G., Nalisnick, E., Rezende, D. J., Mohamed, S., &
  Lakshminarayanan, B. (2021). Normalizing flows for probabilistic modeling and
  inference. *Journal of Machine Learning Research, 22*(57), 1-64.
- Rezende, D. J., & Mohamed, S. (2015). Variational inference with normalizing
  flows. *Proceedings of Machine Learning Research, 37*, 1530-1538.

## Report and interactive site

The publication-grade report is available from the **Read the Report** button on
the GitHub Pages laboratory in `docs/index.html`.

Rebuild the report figures, tables, and editable document with:

```bash
python report/reproduce_report_figures.py
python report/build_report.py
```

The committed PDF is exactly 25 letter-sized pages. Figure and table values are
analytic or deterministic toy-design benchmarks; they are not presented as
measurements of physical Hawking radiation.

## License

MIT. See [LICENSE](LICENSE). Scientific results should cite [CITATION.cff](CITATION.cff).
