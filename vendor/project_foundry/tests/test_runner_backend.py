import sys
import os
import subprocess
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from simple_adp.runner import backend


@pytest.fixture(autouse=True)
def reset_backend_state():
    backend.reset_backend_cache()
    yield
    backend.reset_backend_cache()


def test_cpu_and_gpu_admission_aliases_and_conflicts(monkeypatch):
    monkeypatch.setenv("TRAINING_CONTROL_BACKEND", "cpu")
    assert backend.backend_context() == (True, False, os.environ.get("CUDA_VISIBLE_DEVICES"))
    monkeypatch.delenv("TRAINING_CONTROL_BACKEND")
    monkeypatch.setenv("TABULAR_BACKEND", "gpu")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    assert backend.backend_context() == (False, True, "2")
    monkeypatch.delenv("TABULAR_BACKEND")
    for flag in ("CPU_ONLY", "TRAINING_CONTROL_CPU_ONLY", "OPF_ADP_DISABLE_GPU_ACCELERATORS"):
        monkeypatch.setenv(flag, " YES ")
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
        assert backend.backend_context() == (True, False, "0")
        monkeypatch.delenv(flag)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", " -1 ")
    assert backend.backend_context() == (True, False, " -1 ")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("TRAINING_CONTROL_BACKEND", "gpu")
    monkeypatch.setenv("CPU_ONLY", "1")
    with pytest.raises(RuntimeError, match="conflicting CPU and GPU"):
        backend.backend_context()
    monkeypatch.delenv("CPU_ONLY")
    monkeypatch.setenv("TRAINING_CONTROL_BACKEND", "cpu")
    monkeypatch.setenv("TABULAR_BACKEND", "gpu")
    with pytest.raises(RuntimeError, match="conflicting CPU and GPU"):
        backend.backend_context()


def test_backend_admission_and_device_resolution_use_process_environment_only(monkeypatch):
    monkeypatch.setenv("TRAINING_CONTROL_BACKEND", "cpu")
    assert backend.resolve_torch_device() == "cpu"
    with pytest.raises(TypeError):
        backend.resolve_torch_device({"TRAINING_CONTROL_BACKEND": "gpu"})
    with pytest.raises(TypeError):
        backend.gpu_available({"CPU_ONLY": "1"})


def test_gpu_probe_falls_back_from_torch_to_cupy_and_pins_context(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.delenv("TRAINING_CONTROL_BACKEND", raising=False)
    monkeypatch.delenv("TABULAR_BACKEND", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    class Probe:
        def fill(self, _value):
            return None

    cupy = types.ModuleType("cupy")
    cupy.uint8 = object()
    cupy.empty = lambda *_args, **_kwargs: Probe()
    cupy.asarray = lambda value: ("cupy", value)
    cupy.asnumpy = lambda _value: "converted-to-numpy"
    calls = []
    cupy.cuda = types.SimpleNamespace(runtime=types.SimpleNamespace(
        getDeviceCount=lambda: 1,
        deviceSynchronize=lambda: calls.append("sync"),
    ))
    monkeypatch.setitem(sys.modules, "cupy", cupy)

    assert backend.gpu_available() is True
    assert calls == ["sync"]

    class FakeCuPyArray:
        __module__ = "cupy._core.core"

    monkeypatch.setattr(
        backend, "enable_optional_gpu_accelerators",
        lambda: backend.BackendStatus(True, False, False, True),
    )
    assert backend.array_module(FakeCuPyArray()) is cupy
    assert backend.as_numpy(FakeCuPyArray()) == "converted-to-numpy"
    assert backend.as_cupy("input") == ("cupy", "input")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    with pytest.raises(RuntimeError, match="CUDA context was initialized"):
        backend.gpu_available()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("TRAINING_CONTROL_BACKEND", "cpu")
    with pytest.raises(RuntimeError, match="CUDA context was initialized"):
        backend.resolve_torch_device()


@pytest.mark.parametrize("failure", [
    ImportError("torch cuda import failure"),
    RuntimeError("torch cuda runtime failure"),
    OSError("torch cuda OS failure"),
    AttributeError("torch cuda attribute failure"),
])
@pytest.mark.parametrize("probe_step", ["availability", "allocation", "fill", "synchronize"])
def test_every_documented_torch_probe_failure_falls_back_to_cupy(
    monkeypatch, failure, probe_step,
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.delenv("TRAINING_CONTROL_BACKEND", raising=False)
    monkeypatch.delenv("TABULAR_BACKEND", raising=False)

    def fail(*_args, **_kwargs):
        raise type(failure)(str(failure))

    if probe_step == "availability":
        monkeypatch.setattr(torch.cuda, "is_available", fail)
    else:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        if probe_step == "allocation":
            monkeypatch.setattr(torch, "empty", fail)
        elif probe_step == "fill":
            monkeypatch.setattr(torch, "empty", lambda *_args, **_kwargs: types.SimpleNamespace(fill_=fail))
        else:
            monkeypatch.setattr(torch, "empty", lambda *_args, **_kwargs: types.SimpleNamespace(fill_=lambda _value: None))
            monkeypatch.setattr(torch.cuda, "synchronize", fail)

    class Probe:
        def fill(self, _value):
            return None

    cupy = types.ModuleType("cupy")
    cupy.uint8 = object()
    cupy.empty = lambda *_args, **_kwargs: Probe()
    cupy.cuda = types.SimpleNamespace(runtime=types.SimpleNamespace(
        getDeviceCount=lambda: 1,
        deviceSynchronize=lambda: None,
    ))
    monkeypatch.setitem(sys.modules, "cupy", cupy)

    assert backend.gpu_available() is True


def test_gpu_admission_fails_when_neither_torch_nor_cupy_can_allocate(monkeypatch):
    monkeypatch.setenv("TRAINING_CONTROL_BACKEND", "gpu")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setitem(sys.modules, "cupy", None)

    with pytest.raises(RuntimeError, match="no usable Torch or CuPy"):
        backend.enable_optional_gpu_accelerators()


def test_cpu_admission_blocks_cupy_and_accelerator_tensor_inspection(monkeypatch):
    monkeypatch.setenv("TRAINING_CONTROL_BACKEND", "cpu")

    class FakeCuPyArray:
        __module__ = "cupy._core.core"

    class FakeAcceleratorTensor:
        device = types.SimpleNamespace(type="xpu")

    for value in (FakeCuPyArray(), FakeAcceleratorTensor()):
        with pytest.raises(RuntimeError, match="CPU-admitted worker"):
            backend.as_numpy(value)
        with pytest.raises(RuntimeError, match="CPU-admitted worker"):
            backend.require_cpu_value(value, operation="convert")

    assert backend.status() == backend.BackendStatus(False, False, False, False)


def test_optional_hooks_warn_when_target_packages_were_imported(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.delenv("TRAINING_CONTROL_BACKEND", raising=False)
    monkeypatch.setitem(sys.modules, "pandas", types.ModuleType("pandas"))
    monkeypatch.setitem(sys.modules, "sklearn", types.ModuleType("sklearn"))

    with pytest.warns(RuntimeWarning, match="pandas was already imported"):
        assert backend._try_enable_cudf_pandas() is False
    with pytest.warns(RuntimeWarning, match="target library already imported"):
        assert backend._try_enable_cuml_accel() is False


def test_optional_hooks_install_when_import_order_allows(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.delenv("TRAINING_CONTROL_BACKEND", raising=False)
    for name in ("pandas", "sklearn", "umap", "hdbscan"):
        monkeypatch.delitem(sys.modules, name, raising=False)

    calls = []
    cudf = types.ModuleType("cudf")
    cudf.__path__ = []
    cudf_pandas = types.ModuleType("cudf.pandas")
    cudf_pandas.install = lambda: calls.append("cudf")
    cudf.pandas = cudf_pandas
    cuml = types.ModuleType("cuml")
    cuml.__path__ = []
    cuml_accel = types.ModuleType("cuml.accel")
    cuml_accel.install = lambda: calls.append("cuml")
    cuml.accel = cuml_accel
    monkeypatch.setitem(sys.modules, "cudf", cudf)
    monkeypatch.setitem(sys.modules, "cudf.pandas", cudf_pandas)
    monkeypatch.setitem(sys.modules, "cuml", cuml)
    monkeypatch.setitem(sys.modules, "cuml.accel", cuml_accel)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch, "empty", lambda *_args, **_kwargs: types.SimpleNamespace(fill_=lambda _value: None))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_args: None)
    monkeypatch.setitem(sys.modules, "cupy", None)

    result = backend.enable_optional_gpu_accelerators()

    assert result == backend.BackendStatus(True, True, True, False)
    assert calls == ["cudf", "cuml"]


def test_numpy_pandas_accuracy_and_backend_name_surfaces(monkeypatch):
    monkeypatch.setenv("TRAINING_CONTROL_BACKEND", "cpu")
    assert backend.backend_numpy() is np
    assert backend.backend_pandas().__name__ == "pandas"
    scorer = backend.backend_accuracy_score()
    assert scorer(np.asarray([0, 1]), np.asarray([0, 1])) == 1.0
    logits = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert scorer(labels, logits) == 1.0
    assert list(backend.iter_backend_names()) == [
        "gpu_available=False", "cudf_pandas_enabled=False",
        "cuml_accel_enabled=False", "cupy_available=False",
    ]


def test_seed_resolution_and_cpu_seeding_use_source_precedence(monkeypatch):
    monkeypatch.setenv("TRAINING_CONTROL_BACKEND", "cpu")
    monkeypatch.setenv("OPF_ADP_GLOBAL_SEED", "bad")
    monkeypatch.setenv("SIMPLE_ADP_SEED", "31")
    assert backend.resolve_seed(None) == 42
    assert backend.resolve_seed("bad") == 42
    monkeypatch.delenv("OPF_ADP_GLOBAL_SEED")
    # The source contract has no generic SIMPLE_ADP_SEED fallback.
    assert backend.resolve_seed(None) == 42
    assert backend.resolve_seed(7) == 7
    assert backend.seed_everything(19) == 19
    first = (np.random.randint(10000), torch.rand(1).item())
    backend.seed_everything(19)
    assert (np.random.randint(10000), torch.rand(1).item()) == first


def test_worker_entrypoint_defers_pandas_and_sklearn_until_backend_setup():
    project_root = Path(__file__).parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(project_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    script = (
        "import sys; import simple_adp.runner.worker; "
        "print(int('pandas' in sys.modules), int('sklearn' in sys.modules))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], cwd=project_root, env=env,
        check=True, capture_output=True, text=True,
    )
    assert completed.stdout.strip() == "0 0"
