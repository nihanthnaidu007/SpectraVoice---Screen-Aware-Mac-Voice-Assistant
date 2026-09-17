"""Tests for the --doctor permission and environment checker (Wave 2).

Three layers are covered:
- The report/exit-code machinery, with injected fake probes (fully portable).
- The default check list on Linux (permission probes must SKIP, not error).
- The macOS probe internals, via stub platform modules and a patched
  sys.platform, so the darwin-only branches run under Linux CI.
"""

import io
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

import pytest

from assistant_app import doctor
from assistant_app.doctor import (
    MACOS_FIXES,
    Check,
    CheckState,
    ProbeOutcome,
    build_default_checks,
    doctor_exit_code,
    run_doctor,
)
from assistant_app.utils.config import ConfigManager


def make_check(name, state=CheckState.PASS, detail="detail", fix_it=None):
    return Check(name, lambda: ProbeOutcome(state, detail), fix_it=fix_it)


def run_and_capture(checks):
    stream = io.StringIO()
    results = run_doctor(None, checks=checks, stream=stream)
    return results, stream.getvalue()


class TestReportAndExitCodes:
    def test_all_pass_reports_and_exits_zero(self):
        checks = [make_check("Microphone"), make_check("Screen recording")]
        results, report = run_and_capture(checks)
        assert doctor_exit_code(results) == 0
        assert "Microphone" in report
        assert "Screen recording" in report
        assert "Summary:" in report

    def test_fail_exits_one_and_prints_fix_it_link(self):
        fix = f"System Settings | {MACOS_FIXES['screen'][1]}"
        checks = [make_check("Screen recording", CheckState.FAIL, "denied", fix_it=fix)]
        results, report = run_and_capture(checks)
        assert doctor_exit_code(results) == 1
        assert "Fix:" in report
        assert "Privacy_ScreenCapture" in report

    def test_warn_does_not_fail_the_doctor(self):
        checks = [make_check("Ollama server", CheckState.WARN, "not reachable")]
        results, _ = run_and_capture(checks)
        assert doctor_exit_code(results) == 0

    def test_skip_does_not_fail_the_doctor(self):
        checks = [make_check("Microphone", CheckState.SKIP, "not macOS")]
        results, _ = run_and_capture(checks)
        assert doctor_exit_code(results) == 0

    def test_probe_exception_degrades_to_warn(self):
        def broken():
            raise RuntimeError("probe blew up")

        checks = [Check("Broken probe", broken)]
        results, report = run_and_capture(checks)
        assert results[0].state is CheckState.WARN
        assert doctor_exit_code(results) == 0
        assert "probe error" in report


class TestDefaultChecksOnLinux:
    """On non-macOS the permission probes must skip gracefully — CI depends on it."""

    @pytest.fixture(autouse=True)
    def clean_env_and_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)  # no repo config.yaml — built-in defaults
        for key in ("OPENAI_API_KEY", "TAVILY_API_KEY"):
            monkeypatch.delenv(key, raising=False)

    def test_permission_probes_skip_and_exit_zero(self):
        config_manager = ConfigManager()
        results = run_doctor(config_manager, checks=build_default_checks(config_manager))

        permission_states = {r.name: r.state for r in results[:3]}
        assert set(permission_states) == {"Microphone", "Screen recording", "Accessibility"}
        assert all(state is CheckState.SKIP for state in permission_states.values())
        assert doctor_exit_code(results) == 0

    def test_report_lists_every_default_check(self):
        config_manager = ConfigManager()
        run_doctor(config_manager, checks=build_default_checks(config_manager))
        # run_doctor defaults to stdout when no stream is passed
        # (capsys is not used here — the report above returns results in order).
        names = [r.name for r in build_default_checks(config_manager)]
        assert names == [
            "Microphone",
            "Screen recording",
            "Accessibility",
            "OpenAI API key",
            "Ollama server",
            "Tavily API key",
            "Global hotkeys",
            "Config file",
        ]

    def test_openai_key_fail_only_for_cloud_provider(self, tmp_path, monkeypatch):
        (tmp_path / "config.yaml").write_text("llm:\n  provider: cloud\n")
        config_manager = ConfigManager()
        results = run_doctor(config_manager, checks=build_default_checks(config_manager))
        key_result = next(r for r in results if r.name == "OpenAI API key")
        assert key_result.state is CheckState.FAIL

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        config_manager = ConfigManager()
        results = run_doctor(config_manager, checks=build_default_checks(config_manager))
        key_result = next(r for r in results if r.name == "OpenAI API key")
        assert key_result.state is CheckState.PASS

    def test_openai_key_warn_for_auto_provider(self):
        config_manager = ConfigManager()  # default llm.provider: auto
        results = run_doctor(config_manager, checks=build_default_checks(config_manager))
        key_result = next(r for r in results if r.name == "OpenAI API key")
        assert key_result.state is CheckState.WARN

    def test_openai_key_skipped_for_local_provider(self, tmp_path):
        (tmp_path / "config.yaml").write_text("llm:\n  provider: local\n")
        config_manager = ConfigManager()
        results = run_doctor(config_manager, checks=build_default_checks(config_manager))
        key_result = next(r for r in results if r.name == "OpenAI API key")
        assert key_result.state is CheckState.SKIP


class _StubAVFoundation:
    """Stub for the pyobjc AVFoundation module used by the mic probe."""

    AVMediaTypeAudio = "soun"
    AVAuthorizationStatusAuthorized = 1
    AVAuthorizationStatusNotDetermined = 0
    AVAuthorizationStatusDenied = 2

    auth_status = 1

    class AVCaptureDevice:
        @classmethod
        def authorizationStatusForEntity_(cls, entity):
            return _StubAVFoundation.auth_status


class _StubQuartz:
    preflight_result = True

    @staticmethod
    def CGPreflightScreenCaptureAccess():
        return _StubQuartz.preflight_result


class _StubApplicationServices:
    trusted = True

    @staticmethod
    def AXIsProcessTrusted():
        return _StubApplicationServices.trusted


class _StubPyAudio:
    """Stub for the active PyAudio fallback probe."""

    paInt16 = "int16"

    class PyAudio:
        def open(self, **_kwargs):
            return self

        def read(self, _n):
            return b"\x00" * 32

        def close(self):
            return None

        def terminate(self):
            return None


@pytest.fixture()
def darwin(monkeypatch):
    """Pretend we are on macOS and stub the platform modules."""
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setitem(sys.modules, "AVFoundation", _StubAVFoundation)
    monkeypatch.setitem(sys.modules, "Quartz", _StubQuartz)
    monkeypatch.setitem(sys.modules, "ApplicationServices", _StubApplicationServices)
    monkeypatch.setitem(sys.modules, "pyaudio", _StubPyAudio)


class TestDarwinProbeInternals:
    def test_microphone_authorized_passes(self, darwin):
        _StubAVFoundation.auth_status = 1
        outcome = doctor._probe_microphone()
        assert outcome.state is CheckState.PASS

    def test_microphone_denied_fails(self, darwin):
        _StubAVFoundation.auth_status = 2
        outcome = doctor._probe_microphone()
        assert outcome.state is CheckState.FAIL

    def test_microphone_not_determined_warns(self, darwin):
        _StubAVFoundation.auth_status = 0
        outcome = doctor._probe_microphone()
        assert outcome.state is CheckState.WARN

    def test_microphone_falls_back_to_pyaudio_when_pyobjc_missing(self, darwin, monkeypatch):
        # sys.modules[name] = None makes `import AVFoundation` raise ImportError.
        monkeypatch.setitem(sys.modules, "AVFoundation", None)
        outcome = doctor._probe_microphone()
        assert outcome.state is CheckState.PASS
        assert "microphone" in outcome.detail.lower()

    def test_microphone_pyaudio_failure_fails(self, darwin, monkeypatch):
        monkeypatch.setitem(sys.modules, "AVFoundation", None)

        class FailingPyAudio:
            paInt16 = "int16"

            class PyAudio:
                def open(self, **_kwargs):
                    raise OSError("no default input device")

                def terminate(self):
                    return None

        monkeypatch.setitem(sys.modules, "pyaudio", FailingPyAudio)
        outcome = doctor._probe_microphone()
        assert outcome.state is CheckState.FAIL
        assert "no default input device" in outcome.detail

    def test_screen_preflight_granted_passes(self, darwin):
        _StubQuartz.preflight_result = True
        assert doctor._probe_screen_recording().state is CheckState.PASS

    def test_screen_preflight_denied_fails(self, darwin):
        _StubQuartz.preflight_result = False
        outcome = doctor._probe_screen_recording()
        assert outcome.state is CheckState.FAIL

    def test_accessibility_trusted_passes(self, darwin):
        _StubApplicationServices.trusted = True
        assert doctor._probe_accessibility().state is CheckState.PASS

    def test_accessibility_untrusted_fails(self, darwin):
        _StubApplicationServices.trusted = False
        assert doctor._probe_accessibility().state is CheckState.FAIL

    def test_accessibility_missing_framework_skips(self, darwin, monkeypatch):
        monkeypatch.setitem(sys.modules, "ApplicationServices", None)
        assert doctor._probe_accessibility().state is CheckState.SKIP

    def test_fix_it_links_cover_all_three_permissions(self):
        assert "Privacy_Microphone" in MACOS_FIXES["microphone"][1]
        assert "Privacy_ScreenCapture" in MACOS_FIXES["screen"][1]
        assert "Privacy_Accessibility" in MACOS_FIXES["accessibility"][1]
