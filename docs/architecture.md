# Architecture

## Separation of layers

The code intentionally separates four maps.

1. **Quantum benchmark:** `quantum.py` computes density matrices, von Neumann
   entropy, Page's finite-dimensional expectation, and pure-state fidelity.
2. **Reference measurement model:** `target.py` produces paired secrets and
   accessible radiation coordinates through a frozen conditional bijection.
3. **Statistical density model:** `flow.py` learns a conditional RealNVP density
   with exact change-of-variables accounting.
4. **Inference and evaluation:** `training.py` and `pipeline.py` optimize the
   model and record quantum entropy, classical differential entropy, KL errors,
   and decoder fidelity as distinct columns.

## Data flow

```text
secret s -----------+
                    | visibility v(t)
nuisance epsilon ---+----> accessible y_t ----> frozen scrambler ----> x_t
                                                    target p_t          |
                                                                        v
standard z ----> learned conditional RealNVP q_theta <----------- MLE / VI
                    |
                    +---- exact log q_theta(x_t)
                    +---- inverse coordinate z_hat
                    +---- entropy estimate and decoder fidelity
```

The target scrambler and learned model have the same mathematical class but
independent parameters. This creates a controlled density-estimation problem
without leaking the target parameters into the learned decoder.

## Why paired alignment is explicit

If the base law is spherical Gaussian, then `z` and `Oz` have the same density
for every orthogonal matrix `O`. Therefore maximum likelihood cannot identify a
unique latent orientation. A learned inverse may perfectly model the telemetry
density yet encode the secret in an arbitrary rotation. The optional alignment
term anchors the statistical latent basis to the reference accessible latent.
This is a methodological requirement, not merely a training convenience.

## Numerical stability

- Coupling-layer log-scales are bounded by `scale_clip * tanh(.)`.
- Double precision is the baseline.
- Jacobians are accumulated as log determinants.
- Gradient norms are clipped.
- Mass is stopped above a positive floor.
- Entropies include Monte Carlo standard errors.
- All random generators are explicitly seeded.

## Scaling

An affine coupling layer is linear in the radiation feature dimension outside
its conditioner. Dense conditioner cost is approximately `O(L D H + L H^2)`
per batch element for `L` coupling blocks, dimension `D`, and width `H`.
Replacing dense conditioners with equivariant, sparse, or attention-based blocks
is the intended path to particle sets and larger mode counts.
