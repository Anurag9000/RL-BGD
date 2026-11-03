"""Bounded native worker checks; do not launch research training."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import ModuleType
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_all_training import worker


class FoundryNativeWorkerTest(unittest.TestCase):
    def test_success_exit_and_result_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.json"
            input_path.write_text('{"verified": true}', encoding="utf-8")
            spec = root / "spec.json"
            spec.write_text(json.dumps({
                "run_id": "check-success", "scientific_job": {"id": "json-check"},
                "result_dir": str(root / "results"),
                "command": [sys.executable, "-m", "json.tool", str(input_path)],
            }), encoding="utf-8")
            self.assertEqual(worker(spec), 0)
            self.assertEqual(json.loads((root / "results/result.json").read_text())["status"], "completed")

    def test_failure_does_not_create_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = root / "spec.json"
            spec.write_text(json.dumps({
                "run_id": "check-failure", "scientific_job": {"id": "missing-json"},
                "result_dir": str(root / "results"),
                "command": [sys.executable, "-m", "json.tool", str(root / "missing.json")],
            }), encoding="utf-8")
            self.assertNotEqual(worker(spec), 0)
            self.assertFalse((root / "results/result.json").exists())


    def test_stale_success_is_archived_before_failed_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            results.mkdir()
            (results / "result.json").write_text('{"status": "completed"}')
            (results / "native_process.json").write_text('{"status": "completed"}')
            spec = root / "spec.json"
            spec.write_text(json.dumps({
                "run_id": "failed-retry", "scientific_job": {"id": "stale"},
                "result_dir": str(results),
                "command": [sys.executable, "-m", "json.tool", str(root / "missing.json")],
            }), encoding="utf-8")
            self.assertNotEqual(worker(spec), 0)
            self.assertFalse((results / "result.json").exists())
            archived = results / "attempt_history"
            self.assertEqual(len(list(archived.glob("*result.json"))), 1)
            self.assertEqual(len(list(archived.glob("*native_process.json"))), 1)
            self.assertEqual(
                json.loads((results / "native_process.json").read_text())["status"], "failed"
            )

    def test_zero_exit_without_valid_native_artifacts_is_not_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_dir = root / "results"
            spec = root / "spec.json"
            spec.write_text(json.dumps({
                "run_id": "false-native-success",
                "scientific_job": {"id": "missing_suite__missing_job", "seed": 0},
                "result_dir": str(result_dir),
                "command": [
                    sys.executable, "-m", "foundry_integration.rl_bgd_single",
                    "--suite", "missing_suite", "--job-id", "missing_job",
                    "--seed", "0", "--output-root", str(result_dir / "native"),
                ],
            }), encoding="utf-8")
            native_stub = ModuleType("foundry_integration.rl_bgd_single")

            def reject_native(*_args: object) -> None:
                raise ValueError("invalid canonical native artifacts")

            native_stub.verify_native_job = reject_native  # type: ignore[attr-defined]
            with patch.dict(sys.modules, {"foundry_integration.rl_bgd_single": native_stub}):
                with patch("run_all_training.subprocess.run") as invoked:
                    invoked.return_value.returncode = 0
                    self.assertNotEqual(worker(spec), 0)
            self.assertFalse((result_dir / "result.json").exists())
            state = json.loads((result_dir / "native_process.json").read_text())
            self.assertEqual(state["status"], "failed")
            self.assertIn("invalid canonical native artifacts", state["exception"])


if __name__ == "__main__":
    unittest.main()
