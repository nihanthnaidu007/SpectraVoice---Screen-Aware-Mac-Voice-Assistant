"""W2 S5 tests — transcript export SRT/VTT/Markdown/JSON (zero cloud).

W3 meeting transcripts (append-only JSONL: meta + utterances + explicit gap
markers) gain formatted exports. The serializers are pure (records in, text
out); the only IO is write_transcript_export writing to a destination the
user chose — the acceptance fixtures below pin valid output for all four
formats. Gap policy: SRT/VTT carry utterances only (caption formats);
Markdown and JSON keep the gap markers, because W3's contract is that a
transcript answers "why is there a hole here" — export must not re-introduce
silent loss.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from assistant_app.services import meeting_store as ms
from assistant_app.services import transcript_export as te

SRC = Path(project_root) / "src" / "assistant_app"

START = 1_000_000.0


def _iso(epoch: float) -> str:
    import time

    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(epoch))


def _records() -> list[dict]:
    """Fixture transcript: meta, two utterances, one structured gap between."""
    return [
        {
            "type": "meta",
            "schema_version": 1,
            "language": "english",
            "started_at": START,
            "started_at_iso": _iso(START),
        },
        {
            "type": "utterance",
            "id": 0,
            "text": "Hello team",
            "spoken_at": START + 2.0,
            "spoken_at_iso": _iso(START + 2.0),
            "written_at": START + 3.0,
        },
        {
            "type": "gap",
            "id": 1,
            "started_at": START + 10.0,
            "ended_at": START + 12.5,
            "started_at_iso": _iso(START + 10.0),
            "ended_at_iso": _iso(START + 12.5),
            "reason": "queue_overflow",
        },
        {
            "type": "utterance",
            "id": 2,
            "text": "Back to the agenda — coffre à outils café",
            "spoken_at": START + 20.0,
            "spoken_at_iso": _iso(START + 20.0),
            "written_at": START + 21.0,
        },
    ]


class TestSrt:
    def test_fixture_output_is_valid_srt(self):
        text = te.to_srt(_records())
        blocks = text.strip().split("\n\n")
        assert len(blocks) == 2  # the gap is NOT a caption
        assert blocks[0] == "1\n00:00:02,000 --> 00:00:07,000\nHello team"
        assert blocks[1].startswith("2\n00:00:20,000 --> ")
        assert "Back to the agenda" in blocks[1]

    def test_cue_ends_are_capped_at_five_seconds(self):
        # First cue would naturally run 18s (until the next utterance);
        # captions must never span a gap, so it is capped.
        assert "00:00:02,000 --> 00:00:07,000" in te.to_srt(_records())

    def test_unicode_survives(self):
        text = te.to_srt(_records())
        assert "coffre à outils café" in text

    def test_empty_transcript_is_a_empty_string(self):
        assert te.to_srt([]) == ""

    def test_relative_to_earliest_record_when_meta_is_missing(self):
        records = [r for r in _records() if r["type"] != "meta"]
        text = te.to_srt(records)
        assert text.startswith("1\n00:00:00,000 --> 00:00:05,000\nHello team")


class TestVtt:
    def test_fixture_output_is_valid_vtt(self):
        text = te.to_vtt(_records())
        assert text.startswith("WEBVTT\n")
        assert "00:00:02.000 --> 00:00:07.000" in text
        assert "00:00:20.000 --> " in text
        assert "Hello team" in text

    def test_dot_milliseconds_not_comma(self):
        text = te.to_vtt(_records())
        assert "," not in text.replace("\n", "").split("-->")[0] if "-->" in text else False
        assert "00:00:02.000" in text

    def test_gaps_are_not_cues(self):
        text = te.to_vtt(_records())
        assert "queue_overflow" not in text


class TestMarkdown:
    def test_fixture_output_contains_stamps_utterances_and_gaps(self):
        text = te.to_markdown(_records())
        assert text.startswith("# Meeting transcript")
        assert f"Started: {_iso(START)}" in text
        assert "- **[0:02]** Hello team" in text
        assert "- **[0:20]** Back to the agenda — coffre à outils café" in text
        assert "## Gaps (audio kept but not transcribed)" in text
        assert "- [0:10–0:12] queue_overflow" in text  # 12.5s rounds to 12

    def test_empty_transcript_states_it(self):
        text = te.to_markdown([])
        assert "No transcribed utterances" in text
        assert "Gaps" not in text

    def test_utterance_only_transcript_has_no_gap_section(self):
        records = [r for r in _records() if r["type"] != "gap"]
        assert "Gaps" not in te.to_markdown(records)


class TestJson:
    def test_round_trip_preserves_every_record(self):
        records = _records()
        assert json.loads(te.to_json(records)) == records  # verbatim, incl. gaps

    def test_unicode_is_not_escaped(self):
        assert "coffre à outils café" in te.to_json(_records())


class TestWriteTranscriptExport:
    def _cfg(self, tmp_path):
        return type(
            "Cfg", (), {"audio_dir": str(tmp_path / "corpus"), "retention_hours": 0.0, "language": "english"}
        )()

    def _meeting_with_transcript(self, tmp_path) -> str:
        cfg = self._cfg(tmp_path)
        paths = ms.open_meeting(cfg, START)
        writer = ms.MeetingTranscriptWriter(paths, "english")
        writer.append_utterance(0, "Hello team", START + 2.0)
        ms.write_gap(paths, 1, START + 10.0, START + 12.5, "queue_overflow")
        writer.append_utterance(2, "Back to the agenda", START + 20.0)
        return paths.meeting_id

    def test_writes_the_chosen_format_from_the_stored_transcript(self, tmp_path):
        meeting_id = self._meeting_with_transcript(tmp_path)
        dest = tmp_path / "out"
        out = te.write_transcript_export(self._cfg(tmp_path), meeting_id, str(dest), "srt")
        assert Path(out).exists()
        assert Path(out).name == f"{meeting_id}-transcript.srt"
        stored = ms.read_transcript(str(tmp_path / "corpus" / meeting_id / "transcript.jsonl"))
        assert Path(out).read_text(encoding="utf-8") == te.to_srt(stored)

    def test_all_four_formats_write_and_are_non_empty(self, tmp_path):
        meeting_id = self._meeting_with_transcript(tmp_path)
        dest = tmp_path / "out"
        for fmt, ext in (("srt", ".srt"), ("vtt", ".vtt"), ("markdown", ".md"), ("json", ".json")):
            out = te.write_transcript_export(self._cfg(tmp_path), meeting_id, str(dest), fmt)
            assert Path(out).name == f"{meeting_id}-transcript{ext}"
            assert Path(out).read_text(encoding="utf-8").strip()

    def test_refuses_unknown_format(self, tmp_path):
        meeting_id = self._meeting_with_transcript(tmp_path)
        try:
            te.write_transcript_export(self._cfg(tmp_path), meeting_id, str(tmp_path), "pdf")
        except ValueError as exc:
            assert "pdf" in str(exc) and "srt" in str(exc)
        else:
            raise AssertionError("unknown format must raise ValueError")

    def test_refuses_missing_transcript(self, tmp_path):
        try:
            te.write_transcript_export(self._cfg(tmp_path), "no-such-meeting", str(tmp_path), "srt")
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("missing transcript must raise FileNotFoundError")

    def test_refuses_overwrite(self, tmp_path):
        meeting_id = self._meeting_with_transcript(tmp_path)
        cfg = self._cfg(tmp_path)
        te.write_transcript_export(cfg, meeting_id, str(tmp_path), "json")
        try:
            te.write_transcript_export(cfg, meeting_id, str(tmp_path), "json")
        except FileExistsError:
            pass
        else:
            raise AssertionError("second export must be deliberate — FileExistsError")


class TestZeroCloudAndCliWiring:
    def test_serializer_module_imports_no_network_stack(self):
        source = (SRC / "services" / "transcript_export.py").read_text(encoding="utf-8")
        for forbidden in ("import requests", "import urllib", "import http", "import socket", "urllib.request"):
            assert forbidden not in source

    def test_cli_exposes_the_format_flag_and_dispatches_to_the_exporter(self):
        source = (SRC / "cli.py").read_text(encoding="utf-8")
        assert '"--history-format"' in source
        assert "transcript_export.write_transcript_export(" in source

    def test_directory_copy_path_is_unchanged_without_the_flag(self):
        # --history-export without --history-format keeps copying the meeting
        # directory (the S0 behavior) — the new flag only ADDS a mode.
        source = (SRC / "cli.py").read_text(encoding="utf-8")
        assert "history_ops.export_meeting(meeting_cfg, meeting_id, dest)" in source
