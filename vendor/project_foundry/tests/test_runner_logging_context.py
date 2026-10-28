import logging
import sys
import types

from simple_adp.runner import logging_context


def test_default_resource_context_keeps_cuda_process_count_when_memory_query_fails(monkeypatch):
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(
        is_available=lambda: True,
        current_device=lambda: 0,
        mem_get_info=lambda _device: (_ for _ in ()).throw(RuntimeError("query failed")),
    ))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    context = logging_context.default_resource_context()

    assert context["gpu_processes"] == 1
    assert context["gpu_used_mib"] == 0.0
    assert context["gpu_total_mib"] == 0.0


def test_console_logging_preserves_existing_stream_handler(monkeypatch):
    root = logging.getLogger()
    existing = logging.StreamHandler(sys.stderr)
    monkeypatch.setattr(root, "handlers", [existing])
    monkeypatch.setattr(logging_context, "install_resource_record_factory", lambda: None)

    configured = logging_context.configure_console_logging(level=logging.DEBUG)

    assert configured is root
    assert root.handlers == [existing]
    assert existing.level == logging.NOTSET
    assert root.level == logging.DEBUG


def test_named_console_logger_adds_handler_and_disables_propagation(monkeypatch):
    root = logging.getLogger()
    root_handler = logging.StreamHandler(sys.stderr)
    monkeypatch.setattr(root, "handlers", [root_handler])
    logger = logging.getLogger("simple_adp.test.named_console")
    monkeypatch.setattr(logger, "handlers", [])
    monkeypatch.setattr(logger, "propagate", True)
    monkeypatch.setattr(logging_context, "install_resource_record_factory", lambda: None)

    configured = logging_context.configure_console_logging(
        level=logging.WARNING, name="simple_adp.test.named_console",
    )

    assert configured is logger
    assert logger.propagate is False
    assert logger.level == logging.WARNING
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)
    assert root.handlers == [root_handler]
