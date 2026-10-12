import pytest
import torch


@pytest.fixture(autouse=True)
def restore_torch_state():
    dtype = torch.get_default_dtype()
    with torch.random.fork_rng(devices=[]):
        yield
    torch.set_default_dtype(dtype)
