"""Resource context enrichment for standard Python log records."""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from typing import Any

import psutil

_overlay_provider: Callable[[], dict[str, Any]] = lambda: {}
_factory_installed = False


def default_resource_context() -> dict[str, float | int]:
    context: dict[str, float | int] = {
        "host_total_mib": 0.0,
        "host_used_mib": 0.0,
        "host_available_mib": 0.0,
        "gpu_used_mib": 0.0,
        "gpu_total_mib": 0.0,
        "gpu_processes": 0,
        "cpu_processes": 0,
    }
    try:
        memory = psutil.virtual_memory()
        mib = 1024.0 * 1024.0
        context.update({
            "host_total_mib": memory.total / mib,
            "host_used_mib": memory.used / mib,
            "host_available_mib": memory.available / mib,
        })
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            # The source context reports one GPU-running process as soon as
            # CUDA reports available, even if the subsequent memory query
            # fails and the usage fields remain at their zero defaults.
            context["gpu_processes"] = 1
            free, total = torch.cuda.mem_get_info(torch.cuda.current_device())
            context.update({
                "gpu_used_mib": (total - free) / (1024.0 * 1024.0),
                "gpu_total_mib": total / (1024.0 * 1024.0),
            })
    except Exception:
        pass
    return context


def install_resource_record_factory(provider: Callable[[], dict[str, Any]] | None = None) -> None:
    """Install one resource-aware factory and update its mutable context source."""
    global _factory_installed, _overlay_provider
    if provider is not None:
        _overlay_provider = provider
    if _factory_installed:
        return
    previous = logging.getLogRecordFactory()

    def factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = previous(*args, **kwargs)
        values = default_resource_context()
        try:
            custom = _overlay_provider()
            if isinstance(custom, dict):
                values.update(custom)
        except Exception:
            pass
        for key, value in values.items():
            setattr(record, key, value)
        return record

    logging.setLogRecordFactory(factory)
    _factory_installed = True


def configure_console_logging(
    level: int = logging.INFO,
    *,
    name: str | None = None,
) -> logging.Logger:
    """Configure root or named console logging without replacing handlers."""
    logger = logging.getLogger(name) if name is not None else logging.getLogger()
    logger.setLevel(level)
    if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s "
            "[gpu_processes=%(gpu_processes)s cpu_processes=%(cpu_processes)s "
            "gpu_used_mib=%(gpu_used_mib).1f/"
            "%(gpu_total_mib).1f host_used_mib=%(host_used_mib).1f/"
            "%(host_total_mib).1f]",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(handler)
    if name is not None:
        logger.propagate = False
    install_resource_record_factory()
    return logger
