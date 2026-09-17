"""W4 D1 corpus search tests — scan-on-query over REAL meeting directories.

Corpora are built with the real meeting_store (open_meeting + the JSONL
writer), not mocked trees, so the scanner is proven against the exact
on-disk layout the recorder produces (meta line, utterances, gaps,
summary.md). The 100-meeting smoke test justifies the locked scan-on-query
decision (no derived index, no SQLite/FTS) with a number: a full-corpus
scan must stay in sub-second territory.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timedelta

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from assistant_app.services import history_search
from assistant_app.services import meeting_store as ms


class FakeCfg:
    """The meeting-config surface history scanning needs (FakeCfg pattern)."""

    audio_dir = ""
    language = "english"


def make_cfg(tmp_path) -> FakeCfg:
    cfg = FakeCfg()
    cfg.audio_dir = str(tmp_path / "corpus")
    return cfg


def make_meeting(cfg, utterances, *, started_at=None, gaps=(), summary=None):
    """A REAL meeting directory via the W3 store — meta line already written.

    ``utterances``: list of (seq, text, spoken_at). ``gaps``: (seq, ended_at, reason).
    """
    paths = ms.open_meeting(cfg, started_at if started_at is not None else time.time())
    writer = ms.MeetingTranscriptWriter(paths, "english")
    for seq, text, spoken_at in utterances:
        writer.append_utterance(seq, text, spoken_at)
    for seq, ended_at, reason in gaps:
        writer.append_gap(seq, ended_at - 5.0, ended_at, reason)
    if summary is not None:
        with open(os.path.join(paths.meeting_dir, "summary.md"), "w", encoding="utf-8") as fh:
            fh.write(summary)
    return paths


def noon_before(days_ago: int) -> float:
    """Noon LOCAL time N days ago — far from any midnight boundary so date
    filtering is never flaky against the wall clock."""
    noon = datetime.now().astimezone().replace(hour=12, minute=0, second=0, microsecond=0)
    return (noon - timedelta(days=days_ago)).timestamp()


class TestScanHistory:
    def test_scan_is_empty_without_a_corpus(self, tmp_path):
        rows = history_search.scan_history(make_cfg(tmp_path))
        assert rows == []

    def test_scan_reports_counts_summary_and_order(self, tmp_path):
        cfg = make_cfg(tmp_path)
        old_start = noon_before(3)
        old = make_meeting(
            cfg,
            [(1, "kickoff remarks", old_start + 40.0), (2, "budget talk", old_start + 140.0)],
            started_at=old_start,
            gaps=[(3, old_start + 260.0, "paused")],
            summary="# Old",
        )
        new = make_meeting(cfg, [(1, "only one utterance", noon_before(1) + 500.0)], started_at=noon_before(1))

        rows = history_search.scan_history(cfg)

        assert [r["meeting_id"] for r in rows] == [new.meeting_id, old.meeting_id]  # newest first
        new_row, old_row = rows
        assert new_row["utterance_count"] == 1
        assert new_row["gap_count"] == 0
        assert new_row["summary_present"] is False
        assert old_row["utterance_count"] == 2
        assert old_row["gap_count"] == 1
        assert old_row["summary_present"] is True
        # last event = the gap's end, 260s after the meeting started
        assert old_row["duration_seconds"] == pytest.approx(260.0)
        assert all(r["is_recording"] is False for r in rows)
        assert all(r["matched_snippets"] == [] for r in rows)

    def test_live_meeting_is_marked_and_duration_is_sofar(self, tmp_path):
        cfg = make_cfg(tmp_path)
        paths = make_meeting(cfg, [(1, "in flight", 100.0)], started_at=time.time() - 30.0)
        rows = history_search.scan_history(cfg, live_meeting_ids={paths.meeting_id})
        row = next(r for r in rows if r["meeting_id"] == paths.meeting_id)
        assert row["is_recording"] is True
        assert 0.0 < row["duration_seconds"] < 120.0  # "so far", not anchored to last event

    def test_malformed_lines_are_tolerated(self, tmp_path):
        """Corrupt JSONL lines must not break the scan (tolerant helpers)."""
        cfg = make_cfg(tmp_path)
        paths = make_meeting(cfg, [(1, "before the corruption", 100.0)], started_at=noon_before(2))
        transcript = paths.transcript_path
        with open(transcript, "a", encoding="utf-8") as fh:
            fh.write("{not valid json}\n")
            fh.write(json.dumps({"type": "utterance"}) + "\n")  # missing seq/text fields

        rows = history_search.scan_history(cfg)

        assert len(rows) == 1
        assert rows[0]["utterance_count"] >= 1  # the good line survived
        assert rows[0]["meeting_id"] == paths.meeting_id

    def test_degenerate_directory_still_appears(self, tmp_path):
        """A foreign/crashed directory (no meta line) must be VISIBLE — the
        dashboard shows what is on disk, it does not hide it."""
        cfg = make_cfg(tmp_path)
        root = ms.meeting_root(cfg)
        os.makedirs(os.path.join(root, "20260901-000000-foreign"), exist_ok=True)
        rows = history_search.scan_history(cfg)
        assert [r["meeting_id"] for r in rows] == ["20260901-000000-foreign"]
        assert rows[0]["utterance_count"] == 0
        assert rows[0]["summary_present"] is False

    def test_scan_never_writes(self, tmp_path):
        """Append-only invariant: a scan leaves the corpus byte-identical."""
        cfg = make_cfg(tmp_path)
        make_meeting(cfg, [(1, "do not touch", 100.0)], started_at=noon_before(1), summary="# S")

        def corpus_state():
            state = {}
            for dirpath, _dirnames, filenames in os.walk(ms.meeting_root(cfg)):
                for name in filenames:
                    path = os.path.join(dirpath, name)
                    state[path] = (os.path.getmtime(path), os.path.getsize(path))
            return state

        before = corpus_state()
        history_search.scan_history(cfg)
        history_search.search_history(cfg, "touch")
        assert corpus_state() == before

    def test_date_filtering_after_and_before(self, tmp_path):
        cfg = make_cfg(tmp_path)
        d3 = make_meeting(cfg, [(1, "three days ago", 100.0)], started_at=noon_before(3))
        d1 = make_meeting(cfg, [(1, "yesterday", 100.0)], started_at=noon_before(1))
        d0 = make_meeting(cfg, [(1, "today", 100.0)], started_at=noon_before(0))

        ids = lambda rows: [r["meeting_id"] for r in rows]
        after = time.strftime("%Y-%m-%d", time.localtime(noon_before(2)))
        before = time.strftime("%Y-%m-%d", time.localtime(noon_before(0)))
        assert ids(history_search.scan_history(cfg, after=after)) == [d0.meeting_id, d1.meeting_id]
        # the before-day is INCLUSIVE (bounds run through its 23:59:59)
        assert ids(history_search.scan_history(cfg, before=before)) == [
            d0.meeting_id,
            d1.meeting_id,
            d3.meeting_id,
        ]
        assert ids(history_search.scan_history(cfg, after=after, before=before)) == [
            d0.meeting_id,
            d1.meeting_id,
        ]
        # pinning the upper edge: before = the after-date itself keeps only d3
        assert ids(
            history_search.scan_history(cfg, before=time.strftime("%Y-%m-%d", time.localtime(noon_before(2))))
        ) == [d3.meeting_id]
        assert ids(history_search.scan_history(cfg)) == [d0.meeting_id, d1.meeting_id, d3.meeting_id]

    def test_bad_date_is_a_valueerror(self):
        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            history_search.date_range_bounds("03/2026", None)


class TestSearchHistory:
    def test_finds_utterance_text_case_insensitively(self, tmp_path):
        cfg = make_cfg(tmp_path)
        hit = make_meeting(cfg, [(1, "The BUDGET review starts now", 100.0)], started_at=noon_before(1))
        make_meeting(cfg, [(1, "nothing relevant here", 200.0)], started_at=noon_before(2))

        rows = history_search.search_history(cfg, "budget review")

        assert [r["meeting_id"] for r in rows] == [hit.meeting_id]
        assert rows[0]["matched_snippets"] == ["The BUDGET review starts now"]
        assert "_texts" not in rows[0]  # utterance corpus never leaks past the snippets

    def test_no_match_is_empty_and_empty_query_is_a_caller_bug(self, tmp_path):
        cfg = make_cfg(tmp_path)
        make_meeting(cfg, [(1, "hello", 100.0)], started_at=noon_before(1))
        assert history_search.search_history(cfg, "missing-token") == []
        with pytest.raises(ValueError, match="non-empty"):
            history_search.search_history(cfg, "   ")

    def test_search_respects_date_filters(self, tmp_path):
        cfg = make_cfg(tmp_path)
        old = make_meeting(cfg, [(1, "budget kickoff", 100.0)], started_at=noon_before(5))
        make_meeting(cfg, [(1, "budget wrap-up", 100.0)], started_at=noon_before(1))
        after = time.strftime("%Y-%m-%d", time.localtime(noon_before(2)))
        rows = history_search.search_history(cfg, "budget", after=after)
        assert [r["meeting_id"] for r in rows] != [old.meeting_id]
        assert all("wrap-up" in s for r in rows for s in r["matched_snippets"])

    def test_snippets_are_capped_per_meeting(self, tmp_path):
        cfg = make_cfg(tmp_path)
        utterances = [(seq, f"needle number {seq}", float(seq)) for seq in range(1, 15)]
        meeting = make_meeting(cfg, utterances, started_at=noon_before(1))
        rows = history_search.search_history(cfg, "needle")
        assert [r["meeting_id"] for r in rows] == [meeting.meeting_id]
        assert len(rows[0]["matched_snippets"]) == history_search.SNIPPET_LIMIT


class TestScanOnQueryScale:
    def test_100_meeting_corpus_scans_in_sub_second_time(self, tmp_path):
        """The number that justifies NO derived index (locked decision):
        a 100-meeting corpus — beyond any realistic day of meetings — must
        scan in well under a second, so an index would be pure complexity."""
        cfg = make_cfg(tmp_path)
        base = noon_before(100)
        for day in range(100):
            make_meeting(
                cfg,
                [(1, f"meeting {day} discusses the launch plan", 100.0)],
                started_at=base + day * 86400.0,
            )
        assert len(os.listdir(ms.meeting_root(cfg))) == 100

        started = time.perf_counter()
        rows = history_search.scan_history(cfg)
        elapsed = time.perf_counter() - started

        assert len(rows) == 100
        assert elapsed < 1.0, f"scan took {elapsed:.3f}s — the no-index decision needs revisiting"
