import json
import os
import shutil
import subprocess
from pathlib import Path


def _write_executable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def test_linux_shell_launcher_preserves_thread_settings_and_applies_defaults(tmp_path):
    repository = tmp_path / "repo"
    scripts = repository / "scripts"
    scripts.mkdir(parents=True)
    source_script = Path(__file__).parents[1] / "scripts" / "simple-adp-suite.sh"
    shutil.copyfile(source_script, scripts / source_script.name)

    stub_bin = tmp_path / "stub-bin"
    capture_path = tmp_path / "launch.json"
    calls_path = tmp_path / "calls.txt"
    _write_executable(stub_bin / "nproc", "#!/bin/sh\nprintf '8\\n'\n")
    for command in ("renice", "ionice", "chrt"):
        _write_executable(
            stub_bin / command,
            "#!/bin/sh\nprintf '%s %s\\n' \"$(basename \"$0\")\" \"$*\" >> \"$STUB_CALLS\"\nexit 0\n",
        )

    fake_python = """#!/bin/sh
exec /usr/bin/python3 - "$LAUNCH_CAPTURE" "$@" <<'PY'
import json
import os
import sys
with open(sys.argv[1], "w", encoding="utf-8") as stream:
    json.dump({"env": dict(os.environ), "args": sys.argv[2:]}, stream)
PY
"""
    _write_executable(repository / ".venv" / "bin" / "python", fake_python)

    env = dict(os.environ)
    env.update({
        "PATH": f"{stub_bin}:{env['PATH']}",
        "LAUNCH_CAPTURE": str(capture_path),
        "STUB_CALLS": str(calls_path),
        "OMP_NUM_THREADS": "3",
        "TORCH_INTEROP_THREADS": "4",
        "TABULAR_CHILD_SHARED_CPU": "1",
        "TABULAR_CPU_WORKERS": "6",
    })
    for key in (
        "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "GOTO_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "TORCH_NUM_THREADS",
        "OMP_DYNAMIC", "MKL_DYNAMIC", "OMP_WAIT_POLICY",
    ):
        env.pop(key, None)

    subprocess.run(
        ["bash", str(scripts / source_script.name), "--mode", "suite"],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    captured = json.loads(capture_path.read_text(encoding="utf-8"))
    child_env = captured["env"]
    assert captured["args"] == ["-u", "-m", "simple_adp.runner.launcher", "--mode", "suite"]
    assert child_env["OMP_NUM_THREADS"] == "3"  # caller value is retained
    for key in (
        "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "GOTO_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "TORCH_NUM_THREADS",
    ):
        assert child_env[key] == "8"
    assert child_env["TORCH_INTEROP_THREADS"] == "4"
    assert child_env["OMP_DYNAMIC"] == child_env["MKL_DYNAMIC"] == "FALSE"
    assert child_env["OMP_WAIT_POLICY"] == "ACTIVE"
    assert child_env["TABULAR_CHILD_SHARED_CPU"] == "1"
    assert child_env["TABULAR_CPU_WORKERS"] == "6"
    assert child_env["PYTHONPATH"].split(os.pathsep)[:2] == [str(repository / "src"), str(repository)]

    calls = calls_path.read_text(encoding="utf-8").splitlines()
    assert [line.split(" ", 1)[0] for line in calls] == ["renice", "ionice", "chrt"]
    assert calls[0].startswith("renice -20 -p ")
    assert calls[1].startswith("ionice -c2 -n0 -p ")
    assert calls[2].startswith("chrt -b -p 0 ")
