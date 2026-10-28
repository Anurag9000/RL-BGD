import hashlib
import json

import pytest

from simple_adp.runner import compat
from simple_adp.runner.state import atomic_write_json


def test_environment_and_safe_numeric_helpers_keep_defaults_and_casts():
    env = {"INTEGER": " 7 ", "FLOAT": "1.25", "BAD": "oops"}
    assert compat.env_int("INTEGER", 0, env) == 7
    assert compat.env_float("FLOAT", 0, env) == 1.25
    assert compat.env_int("MISSING", 3.8, env) == 3
    assert compat.env_float("MISSING", 3, env) == 3.0
    assert compat.env_int("BAD", 4, env) == 4
    assert compat.env_float("BAD", 4.5, env) == 4.5
    assert compat.safe_int("9", -1) == 9
    assert compat.safe_int(None, -1) == -1
    assert compat.safe_float("2.5", -1) == 2.5
    assert compat.safe_float(None, -1) == -1.0


def test_csv_and_range_parsers_preserve_recorded_empty_token_behavior():
    assert compat.parse_csv_ints(" 1,, 2, ") == [1, 2]
    with pytest.raises(ValueError):
        compat.parse_csv_ints(None)
    assert compat.parse_range_tuple(" 4, , 9 ") == (4, 9)
    with pytest.raises(ValueError, match="Expected two"):
        compat.parse_range_tuple("4")
    with pytest.raises(ValueError):
        compat.parse_range_tuple("four,9")
    with pytest.raises(ValueError):
        compat.parse_csv_ints("1,nope")


def test_format_and_json_signature_match_canonical_serialization(tmp_path):
    value = {"z": "café", "a": [1, True]}
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    assert compat.json_signature(value) == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert compat.format_kv(True) == "true"
    assert compat.format_kv(False) == "false"
    assert compat.format_kv(None) == "null"
    assert compat.format_kv(5) == "5"
    assert compat.format_kv({"z": 1, "a": 2}) == '{"a": 2, "z": 1}'

    path = tmp_path / "payload.json"
    directory = compat.ensure_dir(tmp_path / "nested" / "created")
    assert directory.is_dir()
    path.write_text('{"ok": true}', encoding="utf-8")
    assert compat.load_json(path) == {"ok": True}
    assert compat.load_json(tmp_path / "missing.json", {"fallback": 1}) == {"fallback": 1}
    path.write_text("not json", encoding="utf-8")
    assert compat.load_json(path, "fallback") == "fallback"
    path.write_bytes(b"\xff\xfe")
    assert compat.load_json(path, "invalid-utf8-fallback") == "invalid-utf8-fallback"


def test_atomic_json_writer_uses_standard_sorted_json_without_string_fallback(tmp_path):
    path = tmp_path / "atomic" / "state.json"
    atomic_write_json(path, {"z": 1, "a": True})
    expected = '{\n  "a": true,\n  "z": 1\n}'
    assert path.read_text(encoding="utf-8") == expected
    with pytest.raises(TypeError):
        atomic_write_json(path, {"unsupported": tmp_path})
    assert path.read_text(encoding="utf-8") == expected
    assert not path.with_suffix(".json.tmp").exists()


def test_atomic_json_replace_failure_propagates_and_keeps_prior_target(tmp_path, monkeypatch):
    from pathlib import Path

    path = tmp_path / "atomic" / "state.json"
    atomic_write_json(path, {"previous": True})
    old_replace = Path.replace

    def fail_replace(source, target):
        if source == path.with_suffix(".json.tmp") and target == path:
            raise OSError("synthetic replace failure")
        return old_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="synthetic replace failure"):
        atomic_write_json(path, {"replacement": True})

    assert json.loads(path.read_text(encoding="utf-8")) == {"previous": True}
    assert json.loads(path.with_suffix(".json.tmp").read_text(encoding="utf-8")) == {"replacement": True}
