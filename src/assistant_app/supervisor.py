"""
Supervisor — auto-restart on unhandled crash.
=============================================

Runs the assistant as a supervised child process. When the child dies from an
unhandled crash (non-zero exit), the supervisor restarts it, preserving the
Wave 0 crash-capture logging for diagnosis:

- The supervisor configures the same rotating log file (`logging:` section of
  config.yaml) via `setup_logging` + `install_crash_handlers`, so its restart
  events and its own crashes are captured too.
- The child's stdout+stderr are appended to the same log file — a child that
  dies before its own logging is set up still leaves its traceback on disk.
- Nothing truncates the log: the W0 crash capture tracebacks stay available.

Crash-loop guard: more than `max_restarts` crashes within `window_seconds`
gives up instead of thrashing forever, leaving a critical record pointing at
the log file. User interrupts (Ctrl+C / SIGTERM reaching the child) are not
crashes — they end supervision with the child's exit status.
"""

import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from assistant_app.utils.config import ConfigManager, SupervisorConfig
from assistant_app.utils.logging_config import get_logger, install_crash_handlers, setup_logging

logger = get_logger(__name__)

# Exit statuses that mean "user interrupted the child", not "child crashed":
# POSIX Popen reports death-by-signal as -N; shells as 128+N.
INTERRUPT_EXIT_CODES = {-signal.SIGINT, -signal.SIGTERM, 128 + signal.SIGINT, 128 + signal.SIGTERM}


@dataclass
class RestartPolicy:
    """When to restart and when to give up."""
    max_restarts: int = 5
    window_seconds: float = 60.0
    backoff_seconds: float = 1.0
    backoff_max_seconds: float = 30.0

    def backoff_for(self, restart_number: int) -> float:
        """Delay before restart #restart_number (1-based): exponential, capped."""
        delay = self.backoff_seconds * (2 ** max(0, restart_number - 1))
        return min(delay, self.backoff_max_seconds)

    @classmethod
    def from_config(cls, cfg: SupervisorConfig) -> "RestartPolicy":
        return cls(
            max_restarts=cfg.max_restarts,
            window_seconds=cfg.window_seconds,
            backoff_seconds=cfg.backoff_seconds,
            backoff_max_seconds=cfg.backoff_max_seconds,
        )


class Supervisor:
    """Restart `command` after crashes, capturing everything in the log file."""

    def __init__(
        self,
        command: list[str],
        policy: RestartPolicy | None = None,
        log_file: str | Path | None = None,
    ):
        self.command = list(command)
        self.policy = policy or RestartPolicy()
        self.log_file = Path(log_file) if log_file else None
        self.logger = get_logger("spectravoice.supervisor")
        self._current_proc: subprocess.Popen | None = None
        self._interrupted = False

    # --- Public API ---------------------------------------------------------

    def run(self) -> int:
        """Run the supervised loop. Returns the child's final exit code.

        - Child exits 0 (clean shutdown, e.g. Ctrl+C handled) → 0.
        - Child died by SIGINT/SIGTERM → that child's exit code, no restart.
        - Crash-loop (more than max_restarts within window_seconds) → 1.
        """
        install_crash_handlers()
        crash_times: deque[float] = deque()
        restart_number = 0

        while True:
            if self._interrupted:
                logger.info("🛑 Supervisor interrupted — stopping with last child status")
                return self._last_exit_code()

            exit_code = self._run_child_once()

            if exit_code == 0:
                logger.info("✅ Child exited cleanly after %d run(s) — supervisor done", restart_number + 1)
                return 0

            if exit_code in INTERRUPT_EXIT_CODES:
                logger.info("🛑 Child exited on user interrupt (code %s) — not restarting", exit_code)
                return exit_code

            now = time.monotonic()
            crash_times.append(now)
            while crash_times and now - crash_times[0] > self.policy.window_seconds:
                crash_times.popleft()

            if len(crash_times) > self.policy.max_restarts:
                logger.critical(
                    "💥 Giving up: child crashed %d times within %.0fs (limit %d restarts). "
                    "Crash tracebacks and output were captured in %s — fix the cause before restarting.",
                    len(crash_times),
                    self.policy.window_seconds,
                    self.policy.max_restarts,
                    self.log_file or "stdout",
                )
                return 1

            restart_number += 1
            backoff = self.policy.backoff_for(restart_number)
            logger.error(
                "💥 Child crashed (exit code %s) — restarting %d/%d in %.1fs; "
                "crash details are captured in %s",
                exit_code,
                restart_number,
                self.policy.max_restarts,
                backoff,
                self.log_file or "stdout",
            )
            time.sleep(backoff)

    def install_signal_forwarding(self) -> None:
        """Forward SIGINT/SIGTERM to the running child (main thread only)."""
        if threading.current_thread() is not threading.main_thread():
            return
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._forward_signal)

    # --- Internals ----------------------------------------------------------

    def _forward_signal(self, signum, _frame) -> None:
        self._interrupted = True
        proc = self._current_proc
        if proc is not None and proc.poll() is None:
            try:
                proc.send_signal(signum)
            except ProcessLookupError:
                pass  # child died between poll() and send_signal

    def _last_exit_code(self) -> int:
        proc = self._current_proc
        if proc is None:
            return 128 + signal.SIGINT
        code = proc.wait()
        return code if code != 0 else 128 + signal.SIGINT

    def _run_child_once(self) -> int:
        """Spawn the child, wait for it to finish, return its exit code."""
        self._interrupted = False
        log_handle = None
        try:
            if self.log_file is not None:
                # Append mode: W0 crash-capture records from earlier runs survive.
                self.log_file.parent.mkdir(parents=True, exist_ok=True)
                # Not a context manager: the handle must stay open for the child's
                # lifetime (the child writes to a dup of this fd); closed in finally.
                log_handle = open(self.log_file, "a", encoding="utf-8", errors="replace")  # noqa: SIM115
                stdout_target: int | None = log_handle.fileno()
                # Line-buffered child output so crash output reaches the log promptly.
                env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            else:
                stdout_target = None
                env = dict(os.environ)

            logger.info("▶️  Starting supervised child: %s", " ".join(self.command))
            self._current_proc = subprocess.Popen(
                self.command,
                stdout=stdout_target,
                stderr=subprocess.STDOUT if stdout_target is not None else None,
                env=env,
            )
            return self._current_proc.wait()
        finally:
            if log_handle is not None:
                log_handle.close()


def supervise(command: list[str], config_manager: ConfigManager) -> int:
    """Wire the supervisor up from the W1 config system and run it."""
    cfg = config_manager.config
    supervisor = Supervisor(
        command=command,
        policy=RestartPolicy.from_config(cfg.supervisor),
        log_file=cfg.logging.file,
    )
    setup_logging(
        debug=cfg.debug,
        log_file=cfg.logging.file,
        max_bytes=cfg.logging.max_size_mb * 1024 * 1024,
        backup_count=cfg.logging.backup_count,
    )
    supervisor.install_signal_forwarding()
    logger.info(
        "🛟 Supervisor active: restarting on crash (max %d restarts / %.0fs); log file: %s",
        cfg.supervisor.max_restarts,
        cfg.supervisor.window_seconds,
        cfg.logging.file,
    )
    return supervisor.run()


def main_entrypoint_command(args: list[str] | None = None) -> list[str]:
    """Build the child argv: this entry point again, supervision disabled.

    Works for both `python main.py` (argv[0] = main.py) and an installed
    console script (argv[0] = the generated launcher, which calls cli.main()).
    """
    if args is None:
        args = sys.argv[1:]
    entry = Path(sys.argv[0]).resolve()
    return [sys.executable, str(entry), *args, "--no-supervisor"]
