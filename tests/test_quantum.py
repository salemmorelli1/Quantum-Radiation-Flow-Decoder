import math

import torch

from quantum_radiation_flow.quantum import (
    page_expected_entropy,
    radiation_density_matrix,
    von_neumann_entropy,
)


def test_bell_state_has_log_two_radiation_entropy() -> None:
    state = torch.tensor(
        [1.0, 0.0, 0.0, 1.0],
        dtype=torch.complex128,
    ) / math.sqrt(2.0)
    rho_r = radiation_density_matrix(state, black_hole_dimension=2, radiation_dimension=2)
    entropy = von_neumann_entropy(rho_r)
    torch.testing.assert_close(entropy, torch.tensor(math.log(2.0)), atol=1.0e-12, rtol=0.0)


def test_page_curve_increases_then_decreases() -> None:
    total = 10
    values = [page_expected_entropy(k, total) for k in range(total + 1)]
    assert values[0] == 0.0
    assert values[-1] == 0.0
    assert values[5] == max(values)
    assert all(a <= b for a, b in zip(values[:5], values[1:6], strict=True))
    assert all(a >= b for a, b in zip(values[5:-1], values[6:], strict=True))
