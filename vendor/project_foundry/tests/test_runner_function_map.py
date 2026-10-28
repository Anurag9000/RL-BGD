from pathlib import Path
import subprocess
import sys


def test_checked_in_runner_function_map_matches_current_ast():
    repository = Path(__file__).parents[1]
    subprocess.run(
        [sys.executable, "scripts/generate_runner_function_map.py", "--check"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
