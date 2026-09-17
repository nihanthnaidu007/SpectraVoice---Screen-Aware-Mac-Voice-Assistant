"""
Doctor — headless first-run diagnostics for SpectraVoice.
=========================================================

`python main.py --doctor` probes the permissions and environment the assistant
depends on and reports each state with a fix-it link, without starting the
assistant or any GUI:

- macOS permissions: microphone, screen recording, accessibility. Each probe
  prefers a precise API status query and falls back to an active probe.
- Environment: OpenAI API key (required only for cloud mode), Ollama
  reachability (optional), Tavily key (optional), config-file health.

Platform behavior: on non-macOS systems (e.g. Linux CI) the permission probes
return SKIP instead of erroring, so the doctor — and the test suite — stays
green cross-platform. macOS outcomes are exercised in tests by injecting fake
probes or stub modules.

Exit code: 0 when no check FAILs, 1 otherwise (WARN/SKIP never fail).
"""

import importlib.util
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from assistant_app.io.hotkeys import parse_hotkey
from assistant_app.utils.config import ConfigManager
from assistant_app.utils.logging_config import get_logger

logger = get_logger(__name__)


class CheckState(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    SKIP = "skip"


_GLYPHS = {
    CheckState.PASS: "✅",
    CheckState.FAIL: "❌",
    CheckState.WARN: "⚠️ ",
    CheckState.SKIP: "⏭️ ",
}


@dataclass
class ProbeOutcome:
    """What a single probe observed."""
    state: CheckState
    detail: str


@dataclass
class Check:
    """A named diagnostic: probe callable plus an optional fix-it hint."""
    name: str
    probe: Callable[[], ProbeOutcome]
    fix_it: str | None = None


@dataclass
class CheckResult:
    """The outcome of running one check."""
    name: str
    state: CheckState
    detail: str
    fix_it: str | None = None


# macOS System Settings panes: (human-readable path, deep link)
MACOS_FIXES = {
    "microphone": (
        "System Settings → Privacy & Security → Microphone → enable your terminal app",
        "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone",
    ),
    "screen": (
        "System Settings → Privacy & Security → Screen Recording → enable your terminal app",
        "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture",
    ),
    "accessibility": (
        "System Settings → Privacy & Security → Accessibility → enable your terminal app",
        "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility",
    ),
}


def _skip_outcome() -> ProbeOutcome:
    return ProbeOutcome(
        CheckState.SKIP,
        f"not macOS (platform={sys.platform}) — hardware permission probe skipped",
    )


# --- Permission probes (macOS real implementations; SKIP elsewhere) ---------

def _probe_microphone() -> ProbeOutcome:
    """Microphone permission: AVFoundation status query, PyAudio open fallback."""
    if sys.platform != "darwin":
        return _skip_outcome()

    try:
        import AVFoundation

        status = AVFoundation.AVCaptureDevice.authorizationStatusForEntity_(
            AVFoundation.AVMediaTypeAudio
        )
        authorized = getattr(AVFoundation, "AVAuthorizationStatusAuthorized", 1)
        not_determined = getattr(AVFoundation, "AVAuthorizationStatusNotDetermined", 0)
        if status == authorized:
            return ProbeOutcome(CheckState.PASS, "Microphone access is authorized for this app")
        if status == not_determined:
            return ProbeOutcome(
                CheckState.WARN,
                "Microphone permission not yet requested — macOS will prompt on first use",
            )
        return ProbeOutcome(CheckState.FAIL, "Microphone access is denied for this app")
    except ImportError:
        pass  # pyobjc-framework-AVFoundation missing — fall through to the active probe

    try:
        import pyaudio

        pa = pyaudio.PyAudio()
        try:
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=44100,
                input=True,
                frames_per_buffer=1024,
            )
            stream.read(1024)
            stream.close()
        finally:
            pa.terminate()
        return ProbeOutcome(CheckState.PASS, "Opened and read the default microphone")
    except Exception as e:
        return ProbeOutcome(CheckState.FAIL, f"Could not open a microphone: {e}")


def _probe_screen_recording() -> ProbeOutcome:
    """Screen Recording permission: Quartz preflight, ImageGrab frame fallback."""
    if sys.platform != "darwin":
        return _skip_outcome()

    try:
        import Quartz

        preflight = getattr(Quartz, "CGPreflightScreenCaptureAccess", None)
        if preflight is not None:
            if preflight():
                return ProbeOutcome(CheckState.PASS, "Screen Recording access is granted")
            return ProbeOutcome(
                CheckState.FAIL,
                "Screen Recording is not granted — screen capture will fail or return wallpaper only",
            )
    except ImportError:
        pass

    try:
        from PIL import ImageGrab

        frame = ImageGrab.grab()
        if frame is None or frame.size == (0, 0):
            return ProbeOutcome(CheckState.FAIL, "Screen capture returned an empty frame")
        return ProbeOutcome(CheckState.PASS, f"Captured a test frame ({frame.size[0]}x{frame.size[1]})")
    except Exception as e:
        return ProbeOutcome(CheckState.FAIL, f"Screen capture probe failed: {e}")


def _probe_accessibility() -> ProbeOutcome:
    """Accessibility permission: AXIsProcessTrusted(); precise API only."""
    if sys.platform != "darwin":
        return _skip_outcome()

    try:
        import ApplicationServices
    except ImportError:
        return ProbeOutcome(
            CheckState.SKIP,
            "pyobjc-framework-ApplicationServices not installed — cannot query Accessibility status",
        )

    if ApplicationServices.AXIsProcessTrusted():
        return ProbeOutcome(CheckState.PASS, "Accessibility access is granted for this app")
    return ProbeOutcome(
        CheckState.FAIL,
        "Accessibility is not granted — keyboard/mouse control will silently fail",
    )


# --- Environment checks (close over config) ---------------------------------

def _check_openai_key(cfg) -> Check:
    def probe() -> ProbeOutcome:
        has_key = bool(os.environ.get("OPENAI_API_KEY"))
        if has_key:
            return ProbeOutcome(CheckState.PASS, "OPENAI_API_KEY is set")
        if cfg.llm.provider == "cloud":
            return ProbeOutcome(CheckState.FAIL, "OPENAI_API_KEY is not set — cloud mode cannot start")
        if cfg.llm.provider == "local":
            return ProbeOutcome(CheckState.SKIP, "OPENAI_API_KEY not set, but local (Ollama) mode is selected")
        return ProbeOutcome(
            CheckState.WARN,
            "OPENAI_API_KEY is not set — cloud mode unavailable; local (Ollama) mode still works",
        )

    return Check(
        "OpenAI API key",
        probe,
        fix_it="Copy .env.example to .env and set OPENAI_API_KEY=sk-... (platform.openai.com)",
    )


def _check_ollama(cfg) -> Check:
    def probe() -> ProbeOutcome:
        import httpx

        url = cfg.llm.ollama_url.rstrip("/")
        try:
            with httpx.Client(timeout=2.0) as client:
                response = client.get(f"{url}/api/tags")
        except Exception as e:
            return ProbeOutcome(
                CheckState.WARN,
                f"Ollama not reachable at {url} ({e.__class__.__name__}) — only needed for --local mode",
            )
        if response.status_code == 200:
            models = response.json().get("models", [])
            return ProbeOutcome(CheckState.PASS, f"Ollama reachable at {url} with {len(models)} model(s)")
        return ProbeOutcome(CheckState.WARN, f"Ollama at {url} returned status {response.status_code}")

    return Check(
        "Ollama server",
        probe,
        fix_it="brew install ollama && ollama serve (macOS), or https://ollama.ai/download",
    )


def _check_tavily_key() -> Check:
    def probe() -> ProbeOutcome:
        if os.environ.get("TAVILY_API_KEY"):
            return ProbeOutcome(CheckState.PASS, "TAVILY_API_KEY is set — web search enabled")
        return ProbeOutcome(CheckState.WARN, "TAVILY_API_KEY not set — web search is disabled (optional)")

    return Check(
        "Tavily API key",
        probe,
        fix_it="Set TAVILY_API_KEY in .env to enable web search (tavily.com)",
    )


def _pynput_available() -> bool:
    """Whether the global-hotkey dependency is installed (graceful probe)."""
    return importlib.util.find_spec("pynput") is not None


def _check_hotkeys(config_manager: ConfigManager) -> Check:
    def probe() -> ProbeOutcome:
        hotkeys = config_manager.config.hotkeys
        specs = {
            "push_to_talk": hotkeys.push_to_talk,
            "dictation_mode": hotkeys.dictation_mode,
            "meeting_toggle": hotkeys.meeting_toggle,
            "mute_toggle": hotkeys.mute_toggle,
            "pause_resume": hotkeys.pause_resume,
            "quit": hotkeys.quit,
        }
        invalid = [name for name, spec in specs.items() if not parse_hotkey(spec)]
        if not _pynput_available():
            return ProbeOutcome(
                CheckState.WARN,
                "pynput is not installed — global hotkeys are disabled; use the HUD menu "
                "and --dictation-mode (VAD dictation works without keys)",
            )
        if invalid:
            return ProbeOutcome(
                CheckState.WARN, f"invalid hotkey config, ignored: {', '.join(invalid)}"
            )
        return ProbeOutcome(
            CheckState.PASS, f"All {len(specs)} global hotkeys parse cleanly (single pynput input path)"
        )

    return Check(
        "Global hotkeys",
        probe,
        fix_it="pip install pynput; grant Accessibility permission (System Settings → Privacy & Security → Accessibility)",
    )


def _check_config(config_manager: ConfigManager) -> Check:
    def probe() -> ProbeOutcome:
        issues = config_manager.validate()
        if issues:
            return ProbeOutcome(CheckState.WARN, f"{len(issues)} config issue(s); first: {issues[0]}")
        source = str(config_manager.config_path) if config_manager.config_path else "built-in defaults (no config file found)"
        return ProbeOutcome(CheckState.PASS, f"Config loads cleanly from {source}")

    return Check("Config file", probe)


def _check_meeting(config_manager: ConfigManager) -> Check:
    """Meeting recording readiness (W3): consent posture, mic TCC, summarizer consistency.

    The kill-switch (``meeting.enabled``) gates everything — when it is off the
    check is a WARN explaining how to opt in, never a FAIL (a disabled feature
    is a valid configuration). When it is on, the same microphone TCC probe the
    assistant depends on runs here (R3: mic permission attaches to the host
    process, so a headless launch fails here instead of mid-meeting).
    """

    def probe() -> ProbeOutcome:
        meeting = config_manager.config.meeting
        if not meeting.enabled:
            return ProbeOutcome(
                CheckState.WARN,
                "Meeting recording is disabled (meeting.enabled: false in config.yaml) — the "
                "kill-switch is engaged: --meeting, the HUD item, and the hotkey all refuse to record",
            )
        notes: list[str] = []
        failures: list[str] = []
        warnings: list[str] = []
        if meeting.summarizer == "cloud" and not meeting.cloud_consent:
            failures.append("summarizer is 'cloud' but cloud_consent is false — summaries refuse to run")
        elif meeting.summarizer == "cloud":
            notes.append("summaries will use Cloud (OpenAI) — third-party speech leaves the device (consented)")
        else:
            notes.append("summaries will use local Ollama (nothing leaves the device)")

        mic = _probe_microphone()
        if mic.state is CheckState.FAIL:
            failures.append(f"microphone unavailable: {mic.detail}")
        elif mic.state is CheckState.WARN:
            warnings.append(mic.detail)
        else:
            notes.append(mic.detail)

        if failures:
            return ProbeOutcome(CheckState.FAIL, "; ".join(failures + warnings))
        if warnings:
            return ProbeOutcome(CheckState.WARN, "; ".join(warnings + notes))
        detail = "; ".join(notes)
        return ProbeOutcome(
            CheckState.PASS,
            f"{detail} — recordings start only via --meeting, the HUD item, or the hotkey "
            f"({config_manager.config.hotkeys.meeting_toggle}), and the recording state stays visible",
        )

    return Check(
        "Meeting recording",
        probe,
        fix_it="Set meeting.enabled: true in config.yaml (kill-switch); recordings still start "
        "per meeting via --meeting / the HUD item / the hotkey",
    )


def build_default_checks(config_manager: ConfigManager) -> list[Check]:
    """The standard doctor check list, in report order."""
    cfg = config_manager.config
    mic_fix, mic_link = MACOS_FIXES["microphone"]
    screen_fix, screen_link = MACOS_FIXES["screen"]
    ax_fix, ax_link = MACOS_FIXES["accessibility"]
    return [
        Check("Microphone", _probe_microphone, fix_it=f"{mic_fix} | {mic_link}"),
        Check("Screen recording", _probe_screen_recording, fix_it=f"{screen_fix} | {screen_link}"),
        Check("Accessibility", _probe_accessibility, fix_it=f"{ax_fix} | {ax_link}"),
        _check_openai_key(cfg),
        _check_ollama(cfg),
        _check_tavily_key(),
        _check_hotkeys(config_manager),
        _check_meeting(config_manager),
        _check_config(config_manager),
    ]


# --- Report ------------------------------------------------------------------

def doctor_exit_code(results: list[CheckResult]) -> int:
    """0 unless any check FAILed (WARN and SKIP are acceptable)."""
    return 1 if any(r.state is CheckState.FAIL for r in results) else 0


def run_doctor(
    config_manager: ConfigManager,
    checks: list[Check] | None = None,
    stream=None,
) -> list[CheckResult]:
    """Run the doctor checks and print a human-readable report.

    Args:
        config_manager: W1 config source (used by the environment checks and
            the default check list).
        checks: Explicit check list (tests inject fakes here); defaults to
            `build_default_checks(config_manager)`.
        stream: Output stream for the report; defaults to stdout.

    Returns:
        One CheckResult per check, in order.
    """
    if checks is None:
        checks = build_default_checks(config_manager)
    if stream is None:
        stream = sys.stdout

    results: list[CheckResult] = []
    for check in checks:
        try:
            outcome = check.probe()
        except Exception as e:
            # A broken probe must never crash the doctor — degrade to WARN.
            logger.debug("Probe %r raised: %s", check.name, e)
            outcome = ProbeOutcome(CheckState.WARN, f"probe error: {e}")
        results.append(CheckResult(check.name, outcome.state, outcome.detail, check.fix_it))

    _print_report(results, stream)
    return results


def _print_report(results: list[CheckResult], stream) -> None:
    width = max((len(r.name) for r in results), default=0) + 3
    print("🩺 SpectraVoice Doctor", file=stream)
    print("=" * 60, file=stream)
    for result in results:
        glyph = _GLYPHS[result.state]
        print(f"{glyph} {result.name:<{width}} {result.detail}", file=stream)
        if result.state is CheckState.FAIL and result.fix_it:
            print(f"   Fix: {result.fix_it}", file=stream)
        elif result.state is CheckState.WARN and result.fix_it:
            print(f"   Hint: {result.fix_it}", file=stream)
    counts = {state: 0 for state in CheckState}
    for result in results:
        counts[result.state] += 1
    summary = ", ".join(f"{counts[s]} {s.value}" for s in CheckState if counts[s])
    print("=" * 60, file=stream)
    print(f"Summary: {len(results)} checks — {summary}", file=stream)
    if counts[CheckState.FAIL]:
        print("Some checks failed — fix the items marked ❌ and re-run --doctor.", file=stream)
    else:
        print("No blocking issues found — you're ready to run `python main.py`.", file=stream)
