"""Backend admission, optional accelerator hooks, and array utilities.

The control flow follows the coordinated-runner ledger. CUDA-adjacent packages
remain optional and are imported lazily, so CPU installs do not need CuPy, cuDF,
or cuML.
"""

from __future__ import annotations

import os
import random
import sys
import warnings
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch

GLOBAL_SEED_ENV = "OPF_ADP_GLOBAL_SEED"
CPU_FLAGS = (
    "CPU_ONLY", "TRAINING_CONTROL_CPU_ONLY", "OPF_ADP_DISABLE_GPU_ACCELERATORS",
    # Generic aliases used by simple-adp's launcher.
    "TABULAR_CPU_ONLY", "SIMPLE_ADP_DISABLE_GPU_ACCELERATORS",
)
CPU_TRUE = {"1", "true", "yes", "on", "y"}


@dataclass(frozen=True, slots=True)
class BackendStatus:
    gpu_available: bool
    cudf_pandas_enabled: bool
    cuml_accel_enabled: bool
    cupy_available: bool


_STATUS: BackendStatus | None = None
_STATUS_CONTEXT: tuple[bool, bool, str | None] | None = None
_PROBED_GPU_CONTEXT: tuple[bool, bool, str | None] | None = None


def backend_context() -> tuple[bool, bool, str | None]:
    """Read admission from this worker's process environment, as the source does."""
    values = os.environ
    assigned_backends = {
        values.get(name, "").strip().lower()
        for name in ("TRAINING_CONTROL_BACKEND", "TABULAR_BACKEND")
    }
    visible = values.get("CUDA_VISIBLE_DEVICES")
    cpu = "cpu" in assigned_backends or any(
        values.get(name, "").strip().lower() in CPU_TRUE for name in CPU_FLAGS
    )
    cpu = cpu or (visible is not None and visible.strip() in {"", "-1"})
    gpu = "gpu" in assigned_backends
    if cpu and gpu:
        raise RuntimeError("conflicting CPU and GPU scheduler admission")
    return cpu, gpu, visible


def _check_admission() -> bool:
    cpu, _gpu, _visible = backend_context()
    context = (cpu, _gpu, _visible)
    if _PROBED_GPU_CONTEXT is not None and _PROBED_GPU_CONTEXT != context:
        raise RuntimeError(
            "CUDA context was initialized before admission/visibility changed; "
            "restart the worker with the new backend environment"
        )
    if (
        _STATUS is not None
        and _STATUS_CONTEXT != context
        and (_STATUS.gpu_available or _STATUS.cudf_pandas_enabled or _STATUS.cuml_accel_enabled or _STATUS.cupy_available)
    ):
        raise RuntimeError(
            "accelerator admission/visibility changed after GPU initialization; "
            "restart the worker with the new backend environment"
        )
    return cpu


def _reject_masked_gpu_array(value: Any) -> None:
    if not _check_admission():
        return
    module = getattr(getattr(value, "__class__", None), "__module__", "")
    if module.startswith("cupy"):
        raise RuntimeError("CPU-admitted worker cannot inspect or convert a CuPy array")
    device = getattr(value, "device", None)
    if getattr(device, "type", None) in {"cuda", "mps", "xpu", "hip"}:
        raise RuntimeError("CPU-admitted worker cannot inspect an accelerator tensor")


def resolve_seed(seed: int | None = None) -> int:
    if seed is not None:
        try:
            return int(seed)
        except Exception:
            pass
    raw = os.environ.get(GLOBAL_SEED_ENV)
    if raw is not None and str(raw).strip():
        try:
            return int(raw)
        except Exception:
            pass
    return 42


def _try_import_cupy() -> Any | None:
    if _check_admission():
        return None
    try:
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() < 1:
            return None
        probe = cp.empty((1,), dtype=cp.uint8)
        probe.fill(1)
        cp.cuda.runtime.deviceSynchronize()
        del probe
        global _PROBED_GPU_CONTEXT
        _PROBED_GPU_CONTEXT = backend_context()
        return cp
    except Exception:
        return None


def gpu_available() -> bool:
    """Probe a real allocation/synchronization; visibility alone is insufficient."""
    global _PROBED_GPU_CONTEXT
    if _check_admission():
        return False
    try:
        if torch.cuda.is_available():
            probe = torch.empty((1,), device="cuda")
            probe.fill_(1)
            torch.cuda.synchronize("cuda")
            del probe
            _PROBED_GPU_CONTEXT = backend_context()
            return True
    except (ImportError, RuntimeError, OSError, AttributeError):
        # A CPU-only Torch wheel does not preclude a working CuPy runtime.
        pass
    return _try_import_cupy() is not None


def _try_enable_cudf_pandas() -> bool:
    if _check_admission():
        return False
    if "pandas" in sys.modules:
        warnings.warn("cuDF pandas unavailable: pandas was already imported", RuntimeWarning, stacklevel=2)
        return False
    try:
        import cudf.pandas as accelerator

        accelerator.install()
        return True
    except ImportError:
        return False
    except Exception as exc:
        warnings.warn(f"cuDF pandas activation failed: {exc}", RuntimeWarning, stacklevel=2)
        return False


def _try_enable_cuml_accel() -> bool:
    if _check_admission():
        return False
    if any(name in sys.modules for name in ("sklearn", "umap", "hdbscan")):
        warnings.warn("cuML accelerator unavailable: target library already imported", RuntimeWarning, stacklevel=2)
        return False
    try:
        import cuml.accel as accelerator

        accelerator.install()
        return True
    except ImportError:
        return False
    except Exception as exc:
        warnings.warn(f"cuML accelerator activation failed: {exc}", RuntimeWarning, stacklevel=2)
        return False


def enable_optional_gpu_accelerators() -> BackendStatus:
    global _STATUS, _STATUS_CONTEXT
    forced_cpu = _check_admission()
    context = backend_context()
    if _STATUS is not None and _STATUS_CONTEXT == context:
        return _STATUS
    if _STATUS is not None and _STATUS_CONTEXT != context:
        if _STATUS.gpu_available or _STATUS.cudf_pandas_enabled or _STATUS.cuml_accel_enabled or _STATUS.cupy_available:
            raise RuntimeError(
                "accelerator admission/visibility changed after GPU initialization; "
                "restart the worker with the new backend environment"
            )
        _STATUS = None
    if forced_cpu:
        _STATUS = BackendStatus(False, False, False, False)
        _STATUS_CONTEXT = context
        return _STATUS

    available = gpu_available()
    if context[1] and not available:
        raise RuntimeError("GPU-admitted child has no usable Torch or CuPy CUDA runtime")
    cudf_enabled = _try_enable_cudf_pandas() if available else False
    cuml_enabled = _try_enable_cuml_accel() if available else False
    cupy_available = _try_import_cupy() is not None if available else False
    _STATUS = BackendStatus(available, cudf_enabled, cuml_enabled, cupy_available)
    _STATUS_CONTEXT = context
    return _STATUS


def status() -> BackendStatus:
    return enable_optional_gpu_accelerators()


def backend_pandas() -> Any:
    enable_optional_gpu_accelerators()
    import pandas as pd

    return pd


def backend_numpy() -> Any:
    backend_status = enable_optional_gpu_accelerators()
    cp = _try_import_cupy() if backend_status.cupy_available else None
    return cp if cp is not None else np


def array_module(*arrays: Any) -> Any:
    for array in arrays:
        _reject_masked_gpu_array(array)
    cp = _try_import_cupy()
    if cp is None:
        return np
    if any(getattr(getattr(array, "__class__", None), "__module__", "").startswith("cupy") for array in arrays if array is not None):
        return cp
    return np


def as_numpy(array: Any) -> Any:
    _reject_masked_gpu_array(array)
    cp = _try_import_cupy()
    if cp is not None and getattr(getattr(array, "__class__", None), "__module__", "").startswith("cupy"):
        return cp.asnumpy(array)
    return np.asarray(array)


def as_cupy(array: Any) -> Any:
    _reject_masked_gpu_array(array)
    backend_status = enable_optional_gpu_accelerators()
    cp = _try_import_cupy() if backend_status.cupy_available else None
    return cp.asarray(array) if cp is not None else np.asarray(array)


def resolve_torch_device() -> str:
    _check_admission()
    context = backend_context()
    if context[0]:
        return "cpu"
    if context[1]:
        if not gpu_available():
            raise RuntimeError("GPU-admitted worker has no usable Torch or CuPy CUDA runtime")
        return "cuda"
    raise RuntimeError("worker environment does not explicitly admit CPU or GPU")


def require_cpu_value(value: Any, *, operation: str) -> None:
    _check_admission()
    if not backend_context()[0]:
        return
    module = getattr(getattr(value, "__class__", None), "__module__", "")
    if module.startswith("cupy"):
        raise RuntimeError(f"CPU-admitted worker cannot {operation} a CuPy array")
    device = getattr(value, "device", None)
    if getattr(device, "type", None) in {"cuda", "mps", "xpu", "hip"}:
        raise RuntimeError(f"CPU-admitted worker cannot {operation} accelerator-backed values")


def seed_everything(seed: int | None = None) -> int:
    resolved = resolve_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(int(resolved))
    random.seed(resolved)
    np.random.seed(resolved)
    state = enable_optional_gpu_accelerators()
    if state.cupy_available:
        cp = _try_import_cupy()
        if cp is not None:
            cp.random.seed(resolved)
    if backend_context()[0]:
        torch.random.default_generator.manual_seed(resolved)
    else:
        torch.manual_seed(resolved)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(resolved)
    return int(resolved)


def backend_accuracy_score():
    enable_optional_gpu_accelerators()

    def accuracy_score(y_true: Any, y_pred: Any) -> float:
        _reject_masked_gpu_array(y_true)
        _reject_masked_gpu_array(y_pred)
        if hasattr(y_true, "detach") or hasattr(y_pred, "detach"):
            true_t = y_true.detach() if hasattr(y_true, "detach") else torch.as_tensor(y_true)
            pred_t = y_pred.detach() if hasattr(y_pred, "detach") else torch.as_tensor(y_pred)
            if pred_t.ndim > 1 and pred_t.shape[-1] > 1:
                pred_t = pred_t.argmax(dim=-1)
            if true_t.ndim > 1 and true_t.shape[-1] > 1:
                true_t = true_t.argmax(dim=-1)
            return float(true_t.reshape(-1).eq(pred_t.reshape(-1)).to(dtype=torch.float32).mean().item())
        cp = _try_import_cupy()
        if cp is not None and (
            getattr(getattr(y_true, "__class__", None), "__module__", "").startswith("cupy")
            or getattr(getattr(y_pred, "__class__", None), "__module__", "").startswith("cupy")
        ):
            true_a, pred_a = cp.asarray(y_true), cp.asarray(y_pred)
            if pred_a.ndim > 1 and pred_a.shape[-1] > 1:
                pred_a = pred_a.argmax(axis=-1)
            if true_a.ndim > 1 and true_a.shape[-1] > 1:
                true_a = true_a.argmax(axis=-1)
            return float(cp.mean(true_a.reshape(-1) == pred_a.reshape(-1)).item())
        true_a, pred_a = np.asarray(y_true), np.asarray(y_pred)
        if pred_a.ndim > 1 and pred_a.shape[-1] > 1:
            pred_a = pred_a.argmax(axis=-1)
        if true_a.ndim > 1 and true_a.shape[-1] > 1:
            true_a = true_a.argmax(axis=-1)
        return float((true_a.reshape(-1) == pred_a.reshape(-1)).mean())

    return accuracy_score


def iter_backend_names() -> Iterable[str]:
    value = status()
    yield f"gpu_available={value.gpu_available}"
    yield f"cudf_pandas_enabled={value.cudf_pandas_enabled}"
    yield f"cuml_accel_enabled={value.cuml_accel_enabled}"
    yield f"cupy_available={value.cupy_available}"


def reset_backend_cache() -> None:
    """Clear process-local probe state; production workers should not switch devices."""
    global _STATUS, _STATUS_CONTEXT, _PROBED_GPU_CONTEXT
    _STATUS = None
    _STATUS_CONTEXT = None
    _PROBED_GPU_CONTEXT = None
