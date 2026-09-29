"""Device selection and runtime metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class DeviceInfo:
    requested: str
    resolved: str
    cuda_available: bool
    gpu_name: str | None
    cuda_runtime: str | None
    torch_cuda: str | None
    total_vram_bytes: int | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def resolve_device(requested: str = "auto") -> torch.device:
    normalized = requested.lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized == "cuda" or normalized.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA was requested ({requested}) but is not available")
        device = torch.device(normalized)
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index out of range: {device.index}")
        return device
    raise ValueError(f"unsupported device specifier: {requested}")


def get_device_info(requested: str = "auto") -> DeviceInfo:
    device = resolve_device(requested)
    gpu_name: str | None = None
    total_vram: int | None = None
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        gpu_name = props.name
        total_vram = props.total_memory
    return DeviceInfo(
        requested=requested,
        resolved=str(device),
        cuda_available=torch.cuda.is_available(),
        gpu_name=gpu_name,
        cuda_runtime=torch.version.cuda,
        torch_cuda=torch.version.cuda,
        total_vram_bytes=total_vram,
    )
