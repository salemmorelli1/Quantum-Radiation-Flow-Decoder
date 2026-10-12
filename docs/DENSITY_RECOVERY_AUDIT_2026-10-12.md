# Density and recovery audit

Audit date: October 12, 2026. Baseline main: `c7e63afca46b324a7637be2199c82f7bc8871bbf`.
Release: 0.2.0. Scope: simulated conditional RealNVP density fitting and paired toy-state recovery.

## Findings and corrections

| Finding reproduced on the baseline | Correction |
| --- | --- |
| Zero, negative-spectrum, and non-Hermitian matrices produced finite entropy after silent symmetrization/projection. | Validate finite square Hermitian PSD trace-one operators before entropy; allow only dtype-level roundoff repair; use the exact zero-eigenvalue limit. |
| A finite state `[1e308,1e308]` normalized into a zero-trace density matrix. | Scale real/imaginary components before normalization. Small nonzero states remain valid as well. |
| Fractional/bool architecture counts, invalid target settings, and a three-column experiment condition passed configuration checks. | Validate integer/numeric domains and require the experiment's two time-mass condition columns. |
| NaN flow inputs and numeric overflow reached output densities. | Validate finite nonempty floating-point inputs, model dtype/device compatibility, finite transforms/log densities, and optimizer updates. |
| Mixed conditions could be labeled one conditional entropy. | Entropy estimation requires a shared condition. Sampling and reverse-KL objectives retain per-sample conditions. |
| Every finite run returned `PASS`, without acceptance rules, trained ablations, or seed-family inference. | Return `COMPLETED` and explicit scientific status `NOT_ASSESSED`; do not certify research conclusions from execution. |
| Recovery had no MCSE or comparator and no saved trained model. | Add paired random-state null and known-scrambler inverse diagnostics, pointwise MCSEs/approximate normal intervals, reloadable model/target checkpoints, and artifact/source/state fingerprints. |
| Initialization/training reused the same seed and the pipeline mutated caller global dtype/RNG. | Separate six named random streams and restore caller CPU RNG/dtype on success and failure. No CUDA execution is claimed. |
| Browser fidelity used visibility to the power 1.35 while the report used power 2; the illustrative score was presented as paired state overlap. | Use the same squared-visibility base proxy and label the browser animation as illustrative. The proxy is not the expected overlap of normalized correlated Gaussian states. |
| The report described trained ablations and seed intervals as completed and called the full zero-initialized flow an identity. | Mark those comparisons as planned, specify fixed permutations, update scientific labels and front matter, and preserve the 25-page report. |
| CI tested only Linux editable installs and direct dependencies. | Verify built CPU wheels on Linux/Windows and Python 3.11/3.12; pin transitive dependencies, check types/browser math/report scope, and replay saved evaluations. |

## Statistical interpretation

Quantum von Neumann/Page entropy, classical differential entropy, and paired
toy-state fidelity remain separate quantities. The target channel stays
`y=v*s+sqrt(1-v^2)*epsilon`, with independent Gaussian secret and nuisance.
Alignment is privileged synthetic training information; density likelihood alone
does not identify a secret-coordinate orientation. The known reference inverse
returns accessible `y`, not the unknown secret or a universal optimal-decoder bound.

The independent random-state null and reference inverse are diagnostics. They do
not substitute for fitted unconditioned and no-alignment ablations. Fidelity
intervals are pointwise approximate normal Monte Carlo intervals conditional on
one trained model. They do not represent seed/retraining uncertainty or simultaneous
coverage of the 21-snapshot curve. Negative Monte Carlo KL means are retained;
they are sampling estimates, not negative population KL divergences.

`physics.page_time` controls the Gaussian visibility midpoint. The separate Haar
benchmark indexes emitted-qubit fraction and peaks at 0.5 for an even qubit total.
Large harmonic numbers use an Euler-Maclaurin expansion accurate to float64 at
the switch threshold, avoiding an exponential summation over Hilbert dimension.
Source/state/file hashes provide integrity checks, not source authentication or
execution attestation. Core simulation fingerprinting uses PyTorch storage and
does not introduce a NumPy dependency into the simulation package.

## Executed default-run evidence

The installed 0.2.0 CPU wheel executed the unchanged baseline configuration:
1,200 optimization steps, batch size 256, eight real coordinates, four complex
state coordinates, 21 evaluation snapshots, 1,024 fresh secrets per snapshot,
float64, root seed 20260824, and one CPU thread. Saved-state replay reproduced
the curve. Full snapshot values and source fingerprints are in
[the audit evidence](AUDIT_BASELINE_EVIDENCE_2026-10-12.json).

At time 1.0:

| Diagnostic | Value |
| --- | ---: |
| Decoder fidelity | 0.208128 |
| Decoder pointwise approximate 95% MC interval | 0.196158 to 0.220098 |
| Independent null fidelity | 0.251103 |
| Null theoretical mean | 0.250000 |
| Paired decoder-minus-null gain | -0.042975 |
| Gain pointwise approximate 95% MC interval | -0.059942 to -0.026007 |
| Known-scrambler inverse fidelity | 0.997563 |
| Forward KL estimate / MCSE (nats) | -0.000158 / 0.002897 |
| Reverse KL estimate / MCSE (nats) | 0.004442 / 0.003011 |
| Maximum coordinate round-trip error | 1.78e-15 |

This single-seed run does not establish successful decoding. Small held-out
density KL estimates can coexist with poor paired recovery. No model budget,
acceptance threshold, or report proxy was tuned to conceal that outcome.

## Verification

- Baseline: 13 existing tests passed; the failures above were reproduced separately.
- Corrected installed wheel: 88 tests, Ruff, strict mypy over all 12 package modules,
  compilation, dependency consistency, and CLI/checkpoint replay passed locally.
- Regression coverage includes nontrivial full-autograd Jacobians, Haar mean versus
  Page expectation, invalid operators, extreme state scaling, Gaussian rotation
  non-identifiability, independent streams, float32 execution, and artifact tampering.
- Node executed the actual browser Page and squared-visibility functions and checked
  the full embedded script syntax. Root/docs pages agree apart from report link paths.
- The corrected editable document was rendered; all 25 letter-sized PDF pages were
  inspected. Caption groups remain with their figures, and report scope markers are
  checked automatically. The report retains the original equation representation.
- GitHub's feature/PR workflows apply the same checks to built wheels on all four
  operating-system/Python combinations. Confirm the PR check results before merging.

## Primary references checked

- [Dinh et al. Real NVP](https://arxiv.org/abs/1605.08803): exact density and sampling via invertible transforms.
- [Page finite-subsystem entropy](https://arxiv.org/abs/gr-qc/9305007): the separate Haar-average benchmark.
- [PyTorch eigvalsh contract](https://docs.pytorch.org/docs/stable/generated/torch.linalg.eigvalsh.html): Hermitian eigenvalue computation; density-operator constraints require explicit validation.
