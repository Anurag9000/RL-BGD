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
from run_all_training import _scientific_run_signature, _source_revision, worker


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

    def test_config_preflight_failure_archives_stale_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            results.mkdir()
            (results / "result.json").write_text('{"status": "completed"}')
            (results / "native_process.json").write_text('{"status": "completed"}')
            (results / "native_config.yaml").write_text("stale config")
            spec = root / "spec.json"
            spec.write_text(json.dumps({
                "run_id": "changed-config",
                "scientific_job": {"id": "example", "config_payload": {"eta": 0.1}},
                "result_dir": str(results),
                "command": [sys.executable, "-m", "json.tool"],
            }))
            yaml_stub = ModuleType("yaml")
            yaml_stub.safe_load = lambda _: {"eta": 0.2}  # type: ignore[attr-defined]
            with patch.dict(sys.modules, {"yaml": yaml_stub}):
                with patch("run_all_training.subprocess.run") as invoked:
                    self.assertNotEqual(worker(spec), 0)
                    invoked.assert_not_called()
            self.assertFalse((results / "result.json").exists())
            history = results / "attempt_history"
            self.assertEqual(len(list(history.glob("*result.json"))), 1)
            self.assertEqual(len(list(history.glob("*native_process.json"))), 1)
            status = json.loads((results / "native_process.json").read_text())
            self.assertEqual(status["status"], "failed")
            self.assertIn("resolved native config changed", status["exception"])

    def test_invalid_argv_preflight_archives_stale_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            results.mkdir()
            (results / "result.json").write_text('{"status": "completed"}')
            spec = root / "spec.json"
            spec.write_text(json.dumps({
                "run_id": "invalid-argv", "scientific_job": {"id": "example"},
                "result_dir": str(results), "command": [sys.executable, 123],
            }))
            with patch("run_all_training.subprocess.run") as invoked:
                self.assertNotEqual(worker(spec), 0)
                invoked.assert_not_called()
            self.assertFalse((results / "result.json").exists())
            self.assertEqual(len(list((results / "attempt_history").glob("*result.json"))), 1)
            status = json.loads((results / "native_process.json").read_text())
            self.assertEqual(status["status"], "failed")
            self.assertIn("nonempty argv array", status["exception"])


    def test_registered_scientific_job_cannot_bypass_native_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = root / "spec.json"
            spec.write_text(json.dumps({
                "run_id": "wrong-wrapper",
                "source_revision": "a" * 40,
                "scientific_job": {
                    "id": "stationary_core__stationary_sac_bgd",
                    "source": "src/rl_bgd/experiments/suites.py:SUITES",
                    "scientific_contract": {"target": "rl_bgd.runners.bgd_sac_lqr:run_bgd_sac_lqr"},
                },
                "result_dir": str(root / "results"),
                "command": [sys.executable, "-m", "json.tool"],
            }))
            with patch("run_all_training.subprocess.run") as invoked:
                self.assertNotEqual(worker(spec), 0)
                invoked.assert_not_called()
            self.assertFalse((root / "results/result.json").exists())
            status = json.loads((root / "results/native_process.json").read_text())
            self.assertIn("canonical wrapper argv", status["exception"])

    def test_disguised_native_module_command_cannot_escape_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = root / "spec.json"
            spec.write_text(json.dumps({
                "run_id": "shifted-module",
                "source_revision": "a" * 40,
                "scientific_job": {"id": "test"},
                "result_dir": str(root / "results"),
                "command": [sys.executable, "-X", "dev", "-m",
                            "foundry_integration.rl_bgd_single"],
            }))
            with patch("run_all_training.subprocess.run") as invoked:
                self.assertNotEqual(worker(spec), 0)
                invoked.assert_not_called()
            status = json.loads((root / "results/native_process.json").read_text())
            self.assertEqual(status["status"], "failed")
            self.assertIn("canonical wrapper argv", status["exception"])


    def test_zero_exit_without_valid_native_artifacts_is_not_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_dir = root / "results"
            spec = root / "spec.json"
            spec.write_text(json.dumps({
                "run_id": "false-native-success",
                "source_revision": "a" * 40,
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
                with patch("run_all_training._source_revision", return_value="a" * 40):
                    with patch("run_all_training.subprocess.run") as invoked:
                        invoked.return_value.returncode = 0
                        self.assertNotEqual(worker(spec), 0)
            self.assertFalse((result_dir / "result.json").exists())
            state = json.loads((result_dir / "native_process.json").read_text())
            self.assertEqual(state["status"], "failed")
            self.assertIn("invalid canonical native artifacts", state["exception"])


    def test_source_revision_changes_outer_scientific_signature(self):
        jobs = [{"id": "same-scientific-job", "seed": 0}]
        first = _scientific_run_signature(jobs, "a" * 40)
        second = _scientific_run_signature(jobs, "b" * 40)
        self.assertEqual(len(first), 64)
        self.assertNotEqual(first, second)
        self.assertEqual(first, _scientific_run_signature(jobs, "a" * 40))
        with self.assertRaises(ValueError):
            _scientific_run_signature(jobs, "not-a-git-commit")

    def test_worker_rejects_changed_revision_before_child_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            results.mkdir()
            (results / "result.json").write_text('{"status": "completed"}')
            spec = root / "spec.json"
            spec.write_text(json.dumps({
                "run_id": "source-changed",
                "source_revision": "a" * 40,
                "scientific_job": {"id": "example"},
                "result_dir": str(results),
                "command": [sys.executable, "-m", "json.tool"],
            }), encoding="utf-8")
            with patch("run_all_training._source_revision", return_value="b" * 40):
                with patch("run_all_training.subprocess.run") as invoked:
                    self.assertNotEqual(worker(spec), 0)
                    invoked.assert_not_called()
            self.assertFalse((results / "result.json").exists())
            result = json.loads((results / "native_process.json").read_text())
            self.assertIn("source revision changed", result["exception"])
            self.assertEqual(len(list((results / "attempt_history").glob("*result.json"))), 1)

    def test_git_dirty_tracked_science_is_rejected(self):
        from types import SimpleNamespace

        responses = [
            SimpleNamespace(returncode=0, stdout="c" * 40),
            SimpleNamespace(returncode=1, stdout=""),
        ]
        with patch("run_all_training.subprocess.run", side_effect=responses):
            with self.assertRaisesRegex(RuntimeError, "uncommitted scientific"):
                _source_revision()



if __name__ == "__main__":
    unittest.main()
