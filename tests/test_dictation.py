"""Tests for dictation mode (W1): cleanup pipeline, VAD machine, config
round-trip, and the DictationController state/cancellation behavior.

Platform-independent by design: transcription and typing are injected fakes,
so no SpeechRecognition, PyAudio, Whisper, pynput, or macOS is required —
mirrors the constraints in tests/test_transcription_worker.py.
"""

import ast
import os
import sys
import threading
import time
from pathlib import Path

import pytest
import yaml

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))


import assistant_app.services.dictation as dictation_module
from assistant_app.io.hotkeys import parse_hotkey
from assistant_app.services.dictation import (
    DictationController,
    TextCleaner,
    VADState,
    VADStateMachine,
    expand_snippets,
    normalize_spacing,
    strip_fillers,
)
from assistant_app.utils.config import ConfigManager


class FakeClock:
    """Deterministic clock for the controller's time.time() calls."""

    def __init__(self, start: float = 1000.0):
        self.t = start

    def time(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def make_cfg(**overrides):
    """A DictationConfig stand-in (avoids importing the full config dataclass
    twice — the real one is exercised in the round-trip tests below)."""
    from assistant_app.utils.config import DictationConfig

    return DictationConfig(**overrides)


def recording_typer():
    """Typer fake recording (text, press_enter) calls."""
    calls: list[tuple[str, bool]] = []

    def typer(text: str, press_enter: bool = False) -> bool:
        calls.append((text, press_enter))
        return True

    return typer, calls


# === TextCleaner pipeline ===


class TestSnippetExpansion:
    def test_spoken_punctuation_expands_through_pipeline(self):
        # Raw expansion leaves punctuation spaced; normalize_spacing glues it.
        cleaner = TextCleaner(snippets={"comma": ",", "period": "."})
        assert cleaner.clean("hello comma world period") == "hello, world."

    def test_raw_expansion_leaves_normalized_spacing_to_the_pipeline(self):
        assert expand_snippets("hello comma world", {"comma": ","}) == "hello , world"

    def test_multiword_snippet(self):
        assert expand_snippets("line one new line line two", {"new line": "\n"}) == "line one \n line two"

    def test_longest_snippet_wins(self):
        snippets = {"new": "N", "new line": "\n"}
        assert expand_snippets("say new line", snippets) == "say \n"

    def test_word_boundary_protects_substrings(self):
        # "comma" inside "command" must never be rewritten
        assert expand_snippets("run the command", {"comma": ","}) == "run the command"

    def test_case_insensitive(self):
        assert expand_snippets("Period", {"period": "."}) == "."


class TestFillerRemoval:
    def test_fillers_removed(self):
        assert strip_fillers("um so this is uh a test", ["um", "uh", "so"]) == "this is a test"

    def test_word_boundary(self):
        # "um" inside "volume" survives
        assert strip_fillers("turn up the volume", ["um"]) == "turn up the volume"

    def test_multiword_filler(self):
        assert strip_fillers("it is you know good", ["you know"]) == "it is good"


class TestTextCleaner:
    def test_standard_style_full_pipeline(self):
        cleaner = TextCleaner(
            snippets={"period": "."},
            filler_words=["um", "uh"],
            filler_removal=True,
            style="standard",
        )
        assert cleaner.clean("um let me think about it period") == "let me think about it."

    def test_minimal_style_keeps_fillers_but_expands_snippets(self):
        cleaner = TextCleaner(
            snippets={"period": "."},
            filler_words=["um"],
            filler_removal=True,
            style="minimal",
        )
        assert cleaner.clean("um ls dash la period") == "um ls dash la."

    def test_spacing_normalized(self):
        assert normalize_spacing("word  ,next .") == "word, next."

    def test_empty_input(self):
        assert TextCleaner().clean("") == ""
        assert TextCleaner().clean("   ") == ""


# === VAD state machine ===


class TestVADStateMachine:
    def test_speech_onset_fires_once(self):
        vad = VADStateMachine(silence_timeout=1.0)
        assert vad.on_speech(10.0) is True  # IDLE -> SPEAKING
        assert vad.on_speech(10.2) is False  # continuation
        assert vad.state == VADState.SPEAKING

    def test_short_silence_does_not_end_utterance(self):
        vad = VADStateMachine(silence_timeout=1.0)
        vad.on_speech(10.0)
        assert vad.on_silence(10.5) is False
        assert vad.state == VADState.TRAILING

    def test_silence_timeout_ends_utterance_once(self):
        vad = VADStateMachine(silence_timeout=1.0)
        vad.on_speech(10.0)
        assert vad.on_silence(10.9) is False
        assert vad.on_silence(11.0) is True
        assert vad.state == VADState.IDLE
        assert vad.on_silence(11.5) is False  # stays idle, no repeat event

    def test_speech_during_trailing_resumes_utterance(self):
        vad = VADStateMachine(silence_timeout=1.0)
        vad.on_speech(10.0)
        vad.on_silence(10.5)  # trailing
        assert vad.on_speech(10.8) is False  # same utterance, not a new onset
        assert vad.state == VADState.SPEAKING

    def test_reset(self):
        vad = VADStateMachine(silence_timeout=1.0)
        vad.on_speech(10.0)
        vad.reset()
        assert vad.state == VADState.IDLE
        assert vad.on_silence(20.0) is False

    def test_invalid_timeout_rejected(self):
        with pytest.raises(ValueError):
            VADStateMachine(silence_timeout=0)


# === Config round-trip ===


class TestDictationConfigRoundTrip:
    def _write_config(self, tmp_path, dictation_section):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump({"mode": "terminal", "dictation": dictation_section}))
        return config_path

    def test_yaml_section_loads_into_typed_config(self, tmp_path):
        path = self._write_config(
            tmp_path,
            {
                "enabled": True,
                "activation": "vad",
                "style": "minimal",
                "app_styles": {"terminal": "standard"},
                "snippets": {"new line": "\n\n"},
                "vad_silence_timeout": 0.9,
                "persist_audio": True,
                "audio_dir": "clips",
            },
        )
        cfg = ConfigManager(path).config.dictation
        assert cfg.enabled is True
        assert cfg.activation == "vad"
        assert cfg.style == "minimal"
        assert cfg.app_styles == {"terminal": "standard"}
        assert cfg.snippets == {"new line": "\n\n"}
        assert cfg.vad_silence_timeout == 0.9
        assert cfg.persist_audio is True
        assert cfg.audio_dir == "clips"

    def test_to_dict_includes_dictation(self, tmp_path):
        path = self._write_config(tmp_path, {"enabled": True, "activation": "vad"})
        exported = ConfigManager(path).to_dict()
        assert exported["dictation"]["enabled"] is True
        assert exported["dictation"]["activation"] == "vad"
        assert exported["dictation"]["persist_audio"] is False

    def test_save_reload_round_trip(self, tmp_path):
        path = self._write_config(tmp_path, {"enabled": True, "vad_silence_timeout": 2.5})
        manager = ConfigManager(path)
        saved_path = tmp_path / "saved.yaml"
        manager.save(saved_path)
        reloaded = ConfigManager(saved_path).config.dictation
        assert reloaded.enabled is True
        assert reloaded.vad_silence_timeout == 2.5

    def test_unknown_keys_ignored(self, tmp_path):
        path = self._write_config(tmp_path, {"enabled": True, "future_key": 1})
        assert ConfigManager(path).config.dictation.enabled is True

    def test_defaults_when_section_missing(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("mode: terminal\n")
        cfg = ConfigManager(config_path).config.dictation
        assert cfg.enabled is False
        assert cfg.activation == "push_to_talk"
        assert cfg.persist_audio is False

    def test_validation_flags_bad_values(self, tmp_path):
        path = self._write_config(tmp_path, {"activation": "hold", "style": "aggressive"})
        issues = ConfigManager(path).validate()
        assert any("dictation.activation" in issue for issue in issues)
        assert any("dictation.style" in issue for issue in issues)

    def test_env_overrides(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VA_DICTATION_ENABLED", "true")
        monkeypatch.setenv("VA_DICTATION_ACTIVATION", "vad")
        path = self._write_config(tmp_path, {})
        cfg = ConfigManager(path).config.dictation
        assert cfg.enabled is True
        assert cfg.activation == "vad"


# === DictationController ===


class TestPushToTalk:
    def test_armed_clip_is_cleaned_and_typed_then_disarms(self):
        typer, calls = recording_typer()
        controller = DictationController(
            make_cfg(style="standard"),
            typer=typer,
            frontmost_app=lambda: "Notes",
        )
        controller.on_ptt_press()
        assert controller.armed is True

        controller.handle_transcript("um hello world")
        controller.wait_for_insertion()

        assert calls == [("hello world", False)]
        assert controller.armed is False  # one hold = one insert

    def test_unarmed_clip_is_ignored(self):
        typer, calls = recording_typer()
        controller = DictationController(make_cfg(), typer=typer, frontmost_app=lambda: "Notes")
        assert controller.handle_transcript("hello") is None
        controller.wait_for_insertion()
        assert calls == []

    def test_release_without_speech_expires_stale_window(self, monkeypatch):
        typer, calls = recording_typer()
        clock = FakeClock()
        monkeypatch.setattr(dictation_module, "time", clock)  # covers press/release too

        controller = DictationController(
            make_cfg(),
            typer=typer,
            frontmost_app=lambda: "Notes",
            stale_release_after=1.0,
        )
        controller.on_ptt_press()
        controller.on_ptt_release()
        clock.advance(2.0)  # release long ago, no clip followed

        controller.handle_transcript("unrelated later speech")
        controller.wait_for_insertion()

        assert calls == []
        assert controller.armed is False

    def test_press_during_insertion_cancels_and_rearms(self):
        # Sentences over 240 chars stay separate chunks (short ones merge),
        # so the cancel can land between chunk 1 and chunk 2.
        long_a = ("alpha " * 50).strip() + "."
        long_b = ("beta " * 50).strip() + "."
        first_chunk = threading.Event()
        release = threading.Event()
        calls: list[tuple[str, bool]] = []

        def blocking_typer(text: str, press_enter: bool = False) -> bool:
            calls.append((text, press_enter))
            if len(calls) == 1:
                first_chunk.set()
                release.wait(timeout=5)
            return True

        controller = DictationController(
            make_cfg(style="standard", insert_enter=True),
            typer=blocking_typer,
            frontmost_app=lambda: "Notes",
        )
        controller.on_ptt_press()
        controller.handle_transcript(f"{long_a} {long_b}")
        assert first_chunk.wait(timeout=5), "insertion never started"

        # A fresh push-to-talk press mid-insertion: cancel the rest, rearm.
        controller.on_ptt_press()
        release.set()
        controller.wait_for_insertion()

        assert calls == [(long_a, False)]  # chunks after the press never typed
        assert controller.armed is True  # restarted capture cleanly

    def test_hallucination_clips_are_ignored(self):
        typer, calls = recording_typer()
        controller = DictationController(make_cfg(), typer=typer, frontmost_app=lambda: "Notes")
        controller.on_ptt_press()
        assert controller.handle_transcript("[inaudible]") is None
        controller.handle_transcript("   ")
        controller.wait_for_insertion()
        assert calls == []


class TestVADMode:
    def test_fragments_join_and_insert_after_silence(self, monkeypatch):
        typer, calls = recording_typer()
        clock = FakeClock()
        monkeypatch.setattr(dictation_module, "time", clock)

        controller = DictationController(
            make_cfg(activation="vad", vad_silence_timeout=1.2),
            typer=typer,
            frontmost_app=lambda: "Notes",
        )
        controller.start()
        assert controller.armed is True  # VAD is continuous

        clock.advance(0.5)
        controller.handle_transcript("hello")
        clock.advance(0.5)  # 0.5s quiet — inside the window
        assert controller.poll_vad() is None  # not ready
        clock.advance(0.4)
        controller.handle_transcript("world")  # speech resumes same utterance
        clock.advance(1.3)  # silence gap closes the utterance
        controller.poll_vad()
        controller.wait_for_insertion()

        assert calls == [("hello world", False)]

    def test_utterance_ready_fires_once_per_gap(self, monkeypatch):
        typer, calls = recording_typer()
        clock = FakeClock()
        monkeypatch.setattr(dictation_module, "time", clock)

        controller = DictationController(
            make_cfg(activation="vad", vad_silence_timeout=0.5),
            typer=typer,
            frontmost_app=lambda: "Notes",
        )
        controller.start()
        clock.advance(0.1)
        controller.handle_transcript("hello")
        clock.advance(1.0)
        controller.poll_vad()
        clock.advance(1.0)
        controller.poll_vad()  # second poll after the same gap: no double insert
        controller.wait_for_insertion()

        assert calls == [("hello", False)]

    def test_mode_toggle_flips_activation_and_rearms(self):
        typer, _calls = recording_typer()
        controller = DictationController(
            make_cfg(activation="vad"),
            typer=typer,
            frontmost_app=lambda: "Notes",
        )
        controller.start()
        assert controller.armed is True
        controller.on_mode_toggle()
        assert controller.activation == "push_to_talk"
        assert controller.armed is False  # PTT waits for a hold
        controller.on_mode_toggle()
        assert controller.activation == "vad"
        assert controller.armed is True


class TestPerAppStyles:
    def test_terminal_style_override_is_minimal(self):
        typer, calls = recording_typer()
        controller = DictationController(
            make_cfg(style="standard"),
            typer=typer,
            frontmost_app=lambda: "Terminal",
        )
        controller.insert("um ls dash la period")
        assert calls == [("um ls dash la.", False)]  # "um" kept in terminals

    def test_unknown_app_uses_default_style(self):
        typer, calls = recording_typer()
        controller = DictationController(
            make_cfg(style="standard"),
            typer=typer,
            frontmost_app=lambda: "Notes",
        )
        controller.insert("um hello period")
        assert calls == [("hello.", False)]

    def test_frontmost_app_failure_falls_back_to_default_style(self):
        typer, calls = recording_typer()

        def broken():
            raise RuntimeError("no display")

        controller = DictationController(make_cfg(style="standard"), typer=typer, frontmost_app=broken)
        controller.insert("um hello period")
        assert calls == [("hello.", False)]


class TestInsertionContract:
    def test_insert_enter_fires_on_last_chunk_only(self):
        # Two long sentences (>240 chars each) stay separate chunks.
        long_a = ("alpha " * 50).strip() + "."
        long_b = ("beta " * 50).strip() + "."
        typer, calls = recording_typer()
        controller = DictationController(
            make_cfg(insert_enter=True),
            typer=typer,
            frontmost_app=lambda: "Notes",
        )
        controller.insert(f"{long_a} {long_b}")
        assert [press_enter for _, press_enter in calls] == [False, True]

    def test_short_sentences_merge_into_single_chunk(self):
        # Sub-240-char transcripts merge: one automation round-trip, no
        # per-sentence latency. Cancellation still works because a fresh
        # press during the (single) insert aborts the remainder via the
        # per-insertion event checked before each chunk.
        controller = DictationController(
            make_cfg(),
            typer=lambda text, press_enter=False: True,
            frontmost_app=lambda: "Notes",
        )
        assert controller._chunk_for_insertion("One. Two. Three.") == ["One. Two. Three."]
        long_pair = ("alpha " * 50).strip() + ". " + ("beta " * 50).strip() + "."
        chunks = controller._chunk_for_insertion(long_pair)
        assert len(chunks) == 2

    def test_typing_failure_stops_insertion(self):
        def failing_typer(text: str, press_enter: bool = False) -> bool:
            return False

        controller = DictationController(make_cfg(), typer=failing_typer, frontmost_app=lambda: "Notes")
        assert controller.insert("hello world") is None

    def test_typer_exception_is_contained(self):
        def exploding_typer(text: str, press_enter: bool = False) -> bool:
            raise RuntimeError("FAILSAFE triggered")

        controller = DictationController(make_cfg(), typer=exploding_typer, frontmost_app=lambda: "Notes")
        assert controller.insert("hello world") is None  # safety-stop surfaced, not swallowed

    def test_on_dictated_callback_fires_on_success(self):
        dictated: list[str] = []
        controller = DictationController(
            make_cfg(),
            typer=lambda text, press_enter=False: True,
            frontmost_app=lambda: "Notes",
            on_dictated=dictated.append,
        )
        controller.insert("hello")
        assert dictated == ["hello"]


class TestAudioRetention:
    def test_default_policy_writes_nothing(self, tmp_path):
        controller = DictationController(
            make_cfg(persist_audio=False, audio_dir=str(tmp_path)),
            typer=lambda text, press_enter=False: True,
        )
        controller.record_audio(b"RIFF-fake-wav")
        assert list(tmp_path.iterdir()) == []  # session-only by default

    def test_persist_audio_writes_wav(self, tmp_path):
        import wave

        controller = DictationController(
            make_cfg(persist_audio=True, audio_dir=str(tmp_path)),
            typer=lambda text, press_enter=False: True,
        )
        controller.record_audio(b"RIFF-fake-data", timestamp=123.456)  # even length: whole 16-bit frames
        files = list(tmp_path.iterdir())
        assert len(files) == 1
        assert files[0].name.startswith("dictation_")
        with wave.open(str(files[0]), "rb") as handle:
            assert handle.getnchannels() == 1
            assert handle.getsampwidth() == 2
            assert handle.getframerate() == 16000
            assert handle.readframes(handle.getnframes()) == b"RIFF-fake-data"

    def test_unsupported_audio_object_ignored(self, tmp_path):
        controller = DictationController(
            make_cfg(persist_audio=True, audio_dir=str(tmp_path)),
            typer=lambda text, press_enter=False: True,
        )
        controller.record_audio(object())
        assert list(tmp_path.iterdir()) == []


class TestClipRetentionSweep:
    """G1 (phase-2 audit): dictation.retention_hours must actually execute.

    Mirrors the H9 meeting tests (tests/test_history_ops.py): the sweep is
    the startup janitor's job — it runs with no dashboard, no controller,
    and no dictation session, and 0 hours keeps clips forever.
    """

    def _aged_clip(self, tmp_path, hours):
        clip = tmp_path / "dictation_1.wav"
        clip.write_bytes(b"x")
        aged = time.time() - hours * 3600.0
        os.utime(clip, (aged, aged))  # file mtime governs the TTL
        return clip

    def test_old_clip_is_swept_when_retention_is_set(self, tmp_path):
        clip = self._aged_clip(tmp_path, hours=10.0)
        assert dictation_module.prune_persisted_clips(str(tmp_path), 1.0) == 1
        assert not clip.exists()

    def test_zero_retention_keeps_clips_forever(self, tmp_path):
        clip = self._aged_clip(tmp_path, hours=10.0)
        assert dictation_module.prune_persisted_clips(str(tmp_path), 0.0) == 0
        assert clip.exists()

    def test_recent_clip_survives_the_sweep(self, tmp_path):
        clip = self._aged_clip(tmp_path, hours=0.1)
        assert dictation_module.prune_persisted_clips(str(tmp_path), 24.0) == 0
        assert clip.exists()

    def test_missing_clip_directory_is_not_an_error(self):
        missing = str(Path("/nonexistent") / "dictation_audio")
        assert dictation_module.prune_persisted_clips(missing, 24.0) == 0

    def test_sweep_runs_headless_no_ui_imports(self):
        """The janitor runs without the dashboard: dictation.py must stay free
        of HUD/AppKit/io imports so the sweep works on headless Linux startup."""
        tree = ast.parse(Path(dictation_module.__file__).read_text(encoding="utf-8"))
        imported = [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
        imported += [
            node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
        ]
        offenders = [
            name
            for name in imported
            if name.startswith(("assistant_app.hud", "assistant_app.io", "AppKit", "Foundation", "Quartz"))
        ]
        assert offenders == []

    def test_orchestrator_startup_sweep_covers_dictation_clips(self):
        """Wiring pin: the orchestrator's startup sweep must call the dictation
        prune. The orchestrator is unimportable on Linux (R9), so the call site
        is checked structurally — the menu-structure precedent."""
        orchestrator = Path(dictation_module.__file__).parents[0] / "spectravoice_assistant.py"
        tree = ast.parse(orchestrator.read_text(encoding="utf-8"))
        called = {
            node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        imported = [
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "assistant_app.services.dictation"
            for alias in node.names
        ]
        assert "prune_persisted_clips" in called
        assert "prune_persisted_clips" in imported


class TestStatusLine:
    def test_status_reflects_mode_and_state(self):
        controller = DictationController(make_cfg(), typer=lambda t, press_enter=False: True)
        assert "push_to_talk" not in controller.status_line()  # human wording
        assert "hold-to-speak" in controller.status_line()
        controller.on_mode_toggle()
        assert "continuous (VAD)" in controller.status_line()


class TestHotkeyParsing:
    def test_aliases_normalize(self):
        assert parse_hotkey("cmd+shift+d") == frozenset({"cmd", "shift", "d"})
        assert parse_hotkey("Command+Shift+D") == frozenset({"cmd", "shift", "d"})
        assert parse_hotkey("ctrl") == frozenset({"ctrl"})
        assert parse_hotkey("option+space") == frozenset({"alt", "space"})

    def test_invalid_specs_rejected(self):
        assert parse_hotkey("") is None
        assert parse_hotkey("cmd++d") is None
        assert parse_hotkey(None) is None

    def test_listener_degrades_gracefully_without_pynput(self):
        """On machines without pynput (Linux CI), start() must return False and
        never raise — dictation continues via CLI flags."""
        try:
            import pynput  # noqa: F401

            pytest.skip("pynput is installed — the degradation path is not exercised here")
        except ImportError:
            pass
        controller = DictationController(make_cfg(), typer=lambda t, press_enter=False: True)
        from assistant_app.io.hotkeys import DictationHotkeyListener

        listener = DictationHotkeyListener(controller, frozenset({"space"}), frozenset({"cmd", "shift", "d"}))
        started = listener.start()
        assert started is False
        listener.stop()  # idempotent
