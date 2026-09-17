"""Tests for the crash-restart supervisor (Wave 2).

Uses real child processes: a script that crashes N times (non-zero exit) and
then exits 0. Verifies restart behavior, the give-up crash-loop guard,
user-interrupt handling, and that child output + restart events are captured
in the log file (building on the Wave 0 logging/crash-capture stack).

POSIX-only in places (signal exit codes); CI and the dev machine are POSIX.
"""

import logging
import os
import signal
import subprocess
import sys
import threading

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

import pytest

from assistant_app.supervisor import (
    INTERRUPT_EXIT_CODES,
    RestartPolicy,
    Supervisor,
    main_entrypoint_command,
)
from assistant_app.utils.config import ConfigManager
from assistant_app.utils.logging_config import setup_logging

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal semantics")


@pytest.fixture()
def fresh_logging():
    """Isolate root-logger handlers and exception hooks per test."""
    saved_hooks = (sys.excepthook, threading.excepthook)
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    sys.excepthook, threading.excepthook = saved_hooks


CRASH_SCRIPT = """\
import os, sys
runs_file = os.environ["SV_RUNS_FILE"]
crash_limit = int(os.environ["SV_CRASH_LIMIT"])
with open(runs_file, "a") as f:
    f.write("run\\n")
count = sum(1 for _ in open(runs_file))
if count <= crash_limit:
    print("child-stderr-sentinel", file=sys.stderr)
    sys.exit(1)
print("child-ok-sentinel")
sys.exit(0)
"""


def make_supervisor(tmp_path, crash_limit, max_restarts=5, log_file=None):
    script = tmp_path / "child.py"
    script.write_text(CRASH_SCRIPT)
    policy = RestartPolicy(
        max_restarts=max_restarts,
        window_seconds=60.0,
        backoff_seconds=0.01,
        backoff_max_seconds=0.05,
    )
    log_path = log_file if log_file is not None else tmp_path / "assistant.log"
    supervisor = Supervisor(
        command=[sys.executable, str(script)],
        policy=policy,
        log_file=str(log_path),
    )
    # Route the supervisor's own log records into the same file the child
    # writes to — mirroring the production wiring (one rotating log).
    setup_logging(log_file=str(log_path))
    return supervisor, log_path


class TestRestartBehavior:
    def test_crashes_are_restarted_until_success(self, tmp_path, fresh_logging, monkeypatch):
        monkeypatch.setenv("SV_CRASH_LIMIT", "2")
        runs_file = tmp_path / "runs.txt"
        monkeypatch.setenv("SV_RUNS_FILE", str(runs_file))

        supervisor, _ = make_supervisor(tmp_path, crash_limit=2)
        exit_code = supervisor.run()

        assert exit_code == 0
        assert len(runs_file.read_text().splitlines()) == 3  # 2 crashes + 1 success

    def test_restart_events_are_logged(self, tmp_path, fresh_logging, monkeypatch):
        monkeypatch.setenv("SV_CRASH_LIMIT", "2")
        monkeypatch.setenv("SV_RUNS_FILE", str(tmp_path / "runs.txt"))

        supervisor, log_path = make_supervisor(tmp_path, crash_limit=2)
        supervisor.run()

        text = log_path.read_text()
        assert "restarting 1/5" in text
        assert "restarting 2/5" in text
        assert "Child exited cleanly" in text

    def test_gives_up_after_max_restarts(self, tmp_path, fresh_logging, monkeypatch):
        monkeypatch.setenv("SV_CRASH_LIMIT", "1000")  # always crashes
        runs_file = tmp_path / "runs.txt"
        monkeypatch.setenv("SV_RUNS_FILE", str(runs_file))

        supervisor, log_path = make_supervisor(tmp_path, crash_limit=1000, max_restarts=2)
        exit_code = supervisor.run()

        assert exit_code == 1
        # 1 initial run + 2 restarts, then the supervisor stops.
        assert len(runs_file.read_text().splitlines()) == 3
        assert "Giving up" in log_path.read_text()

    def test_clean_exit_zero_never_restarts(self, tmp_path, fresh_logging, monkeypatch):
        monkeypatch.setenv("SV_CRASH_LIMIT", "0")
        runs_file = tmp_path / "runs.txt"
        monkeypatch.setenv("SV_RUNS_FILE", str(runs_file))

        supervisor, _ = make_supervisor(tmp_path, crash_limit=0)
        exit_code = supervisor.run()

        assert exit_code == 0
        assert len(runs_file.read_text().splitlines()) == 1


class TestInterruptHandling:
    def test_sigint_death_is_not_restarted(self, tmp_path, fresh_logging, monkeypatch):
        # A child that dies from SIGINT is a user interrupt, not a crash:
        # exactly one spawn, and the child's signal status is returned.
        script = tmp_path / "self_interrupt.py"
        script.write_text(
            "import os, signal\n"
            "signal.signal(signal.SIGINT, signal.SIG_DFL)\n"
            "os.kill(os.getpid(), signal.SIGINT)\n"
        )
        runs_file = tmp_path / "runs.txt"
        runs_file.write_text("")
        monkeypatch.setenv("SV_RUNS_FILE", str(runs_file))

        policy = RestartPolicy(max_restarts=5, backoff_seconds=0.01)
        supervisor = Supervisor(
            command=[sys.executable, str(script)],
            policy=policy,
            log_file=str(tmp_path / "assistant.log"),
        )
        setup_logging(log_file=str(tmp_path / "assistant.log"))

        exit_code = supervisor.run()
        assert exit_code == -signal.SIGINT
        assert exit_code in INTERRUPT_EXIT_CODES

    def test_signal_forwarding_reaches_the_child(self, tmp_path):
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        supervisor = Supervisor(command=["true"])
        supervisor._current_proc = proc
        supervisor._forward_signal(signal.SIGTERM, None)
        assert proc.wait(timeout=5) == -signal.SIGTERM


class TestLogCapture:
    def test_child_output_and_restart_events_share_the_log(
        self, tmp_path, fresh_logging, monkeypatch
    ):
        """The W0 crash-capture contract: a crash leaves its evidence on disk."""
        monkeypatch.setenv("SV_CRASH_LIMIT", "1")
        monkeypatch.setenv("SV_RUNS_FILE", str(tmp_path / "runs.txt"))

        supervisor, log_path = make_supervisor(tmp_path, crash_limit=1)
        supervisor.run()

        text = log_path.read_text()
        # The child's own output (its "crash report") was captured...
        assert "child-stderr-sentinel" in text
        # ...and the supervisor's restart event sits alongside it.
        assert "restarting 1/5" in text

    def test_log_is_appended_to_not_truncated(self, tmp_path, fresh_logging, monkeypatch):
        monkeypatch.setenv("SV_CRASH_LIMIT", "1")
        monkeypatch.setenv("SV_RUNS_FILE", str(tmp_path / "runs.txt"))

        log_path = tmp_path / "assistant.log"
        log_path.write_text("PRIOR-CRASH-EVIDENCE\n")  # e.g. W0 capture from run 1

        supervisor, _ = make_supervisor(tmp_path, crash_limit=1, log_file=log_path)
        supervisor.run()

        text = log_path.read_text()
        assert "PRIOR-CRASH-EVIDENCE" in text, "supervisor must preserve prior logs"
        assert "restarting 1/5" in text


class TestBackoff:
    def test_backoff_doubles_and_is_capped(self):
        policy = RestartPolicy(backoff_seconds=0.5, backoff_max_seconds=2.0)
        assert policy.backoff_for(1) == 0.5
        assert policy.backoff_for(2) == 1.0
        assert policy.backoff_for(3) == 2.0
        assert policy.backoff_for(4) == 2.0  # capped
        assert policy.backoff_for(10) == 2.0


class TestConfigWiring:
    def test_child_command_disables_supervision(self):
        argv = ["--debug", "--voice", "nova"]
        child = main_entrypoint_command(argv)
        assert child[-1] == "--no-supervisor"
        assert "--debug" in child
        assert child[0] == sys.executable

    def test_policy_comes_from_the_config_system(self, tmp_path, monkeypatch):
        (tmp_path / "config.yaml").write_text(
            "supervisor:\n"
            "  enabled: true\n"
            "  max_restarts: 3\n"
            "  window_seconds: 30\n"
            "  backoff_seconds: 2\n"
            "  backoff_max_seconds: 10\n"
        )
        monkeypatch.chdir(tmp_path)
        cfg = ConfigManager().config
        policy = RestartPolicy.from_config(cfg.supervisor)
        assert policy.max_restarts == 3
        assert policy.window_seconds == 30.0
        assert policy.backoff_seconds == 2.0
        assert policy.backoff_max_seconds == 10.0
        assert cfg.supervisor.enabled is True
