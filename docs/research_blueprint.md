# Research Blueprint

## Estimand hierarchy

The primary quantum estimand is the radiation von Neumann entropy

```text
S_R(t) = -Tr[rho_R(t) log rho_R(t)].
```

For a measurement channel with outcomes `X_t`, the flow estimand is

```text
h_theta(t) = -E_q[log q_theta(X_t | t, M_t)].
```

These are equal only under restrictive measurement conditions. The protocol
therefore treats the flow curve as a classical measurement diagnostic and uses
the Page curve as a separate quantum benchmark.

## Conditional normalizing flow

For `x=f_theta(z;c_t)`, `z~N(0,I)`, and `c_t=(t,M_t)`,

```text
log q_theta(x|c_t)
  = log phi(f_theta^{-1}(x;c_t))
    + log |det D_x f_theta^{-1}(x;c_t)|.
```

Data-rich fitting minimizes `-E_p log q`, equivalent to forward KL up to the
unknown target entropy. When a normalized target density is available, the
reverse-KL variational free energy is

```text
F(theta;c_t) = E_q[log q_theta(X|c_t) - log p(X|c_t)].
```

The time-integrated objective is a quadrature-weighted sum over snapshots plus
a paired latent-alignment penalty.

## Confirmatory hypotheses

- H1: The held-out forward-KL estimate is below its preregistered tolerance at
  every time snapshot.
- H2: Round-trip coordinate error and Jacobian cancellation remain near machine
  precision.
- H3: Decoder fidelity increases across the Page-time visibility transition and
  exceeds the unconditioned-flow baseline after Page time.
- H4: A measurement-complete density-operator estimator reproduces the Page
  benchmark within uncertainty; classical flow entropy alone is not used for
  this claim.

## Adversarial tests

- Rotate the Gaussian latent basis to demonstrate non-identifiability.
- Remove time conditioning to test temporal aliasing.
- Remove the alignment penalty to expose density/decoder separation.
- Increase scale clipping to induce Jacobian stiffness.
- Add mode imbalance and assess reverse-KL mode seeking.
- Withhold measurement settings to show phase-information non-identifiability.

## Extension to quantum tomography

A physical experiment requires an informationally complete POVM or multiple
measurement bases. A positive semidefinite trace-one density model can be
parameterized as `rho=L L^dagger / Tr(L L^dagger)` and fitted through Born-rule
likelihoods. The flow can then model measurement outcomes or amortize the
posterior over `L`; it cannot replace the density-operator constraints.
