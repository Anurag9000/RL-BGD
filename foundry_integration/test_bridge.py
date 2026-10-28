"""Bounded native worker checks; do not launch research training."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
