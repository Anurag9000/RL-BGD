import pytest
import torch

from rl_bgd.utils.device import resolve_device


def test_auto_resolves_to_available_backend() -> None:
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert resolve_device("auto").type == expected


def test_invalid_device_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported device"):
        resolve_device("tpu")
