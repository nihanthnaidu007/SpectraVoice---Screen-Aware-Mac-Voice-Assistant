"""Tests for the new CLI flags: --doctor and --no-supervisor (Wave 2)."""

import os
import subprocess
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from assistant_app import cli


class TestFlagParsing:
    def test_doctor_flag_defaults_off(self):
        args = cli.parse_args([])
        assert args.doctor is False

    def test_doctor_flag_is_set(self):
        args = cli.parse_args(["--doctor"])
        assert args.doctor is True

    def test_no_supervisor_flag_defaults_off(self):
        args = cli.parse_args([])
        assert args.no_supervisor is False

    def test_no_supervisor_flag_is_set(self):
        args = cli.parse_args(["--no-supervisor"])
        assert args.no_supervisor is True

    def test_existing_flags_still_parse(self):
        args = cli.parse_args(["--gui", "--debug", "--minimal", "--voice", "nova"])
        assert args.gui is True
        assert args.debug is True
        assert args.minimal is True
        assert args.voice == "nova"


class TestDoctorShortCircuit:
    def test_main_with_doctor_flag_reports_and_exits_cleanly(self, tmp_path, monkeypatch):
        # Empty cwd + no API keys: probes skip (Linux), optional checks warn.
        monkeypatch.chdir(tmp_path)
        for key in ("OPENAI_API_KEY", "TAVILY_API_KEY"):
            monkeypatch.delenv(key, raising=False)

        exit_code = cli.main(["--doctor"])
        assert exit_code == 0


class TestMainPyEndToEnd:
    def test_python_main_py_doctor_runs_headless(self):
        """The documented first-run command: `python main.py --doctor`."""
        repo_root = project_root
        result = subprocess.run(
            [sys.executable, "main.py", "--doctor"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Doctor" in result.stdout
        assert "Summary:" in result.stdout
