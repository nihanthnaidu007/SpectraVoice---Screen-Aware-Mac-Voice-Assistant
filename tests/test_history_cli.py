"""W4 D3 CLI tests — the headless history surface (Linux-testable path).

These run `cli.main` directly: the history flags short-circuit before any
assistant/LLM startup (the --doctor pattern), so the whole list/search/
export/delete surface is exercisable on Linux CI. Destructive refusal
(without --confirm) gets its own exit-code test — the structural gate is
double-checked here through the real argument path.
"""

from __future__ import annotations

import os
import sys
import time

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from assistant_app import cli
from assistant_app.services import meeting_store as ms


class FakeCfg:
    audio_dir = ""  # default corpus root: ./logs/meetings under the cwd
    language = "english"


def make_meeting(text="needle in a transcript"):
    """A REAL meeting directory under the default corpus root (cwd-based)."""
    paths = ms.open_meeting(FakeCfg(), time.time())
    writer = ms.MeetingTranscriptWriter(paths, "english")
    writer.append_utterance(1, text, 100.0)
    return paths


@pytest.fixture(autouse=True)
def isolated_cwd(tmp_path, monkeypatch):
    """Fresh cwd per test: default corpus root + config resolve inside it."""
    monkeypatch.chdir(tmp_path)


class TestFlagParsing:
    def test_history_flags_default_off(self):
        args = cli.parse_args([])
        assert args.history_list is False
        assert args.history_search is None
        assert args.history_after is None
        assert args.history_before is None
        assert args.history_delete is None
        assert args.confirm is False

    def test_history_flags_parse(self):
        args = cli.parse_args(
            ["--history-search", "budget", "--history-after", "2026-09-01", "--history-before", "2026-09-17"]
        )
        assert args.history_search == "budget"
        assert args.history_after == "2026-09-01"
        assert args.history_before == "2026-09-17"

    def test_export_takes_id_and_optional_dir(self):
        args = cli.parse_args(["--history-export", "abc123"])
        assert args.history_export == ["abc123"]
        args = cli.parse_args(["--history-export", "abc123", "/tmp/dest"])
        assert args.history_export == ["abc123", "/tmp/dest"]


class TestHistoryList:
    def test_empty_corpus_lists_nothing(self, capsys):
        assert cli.main(["--history-list"]) == 0
        assert "No stored meetings." in capsys.readouterr().out

    def test_lists_a_real_corpus(self, capsys):
        make_meeting()
        assert cli.main(["--history-list"]) == 0
        out = capsys.readouterr().out
        assert "MEETING" in out and "UTT" in out  # the aligned table header
        assert " 1 " in out  # utterance count column


class TestHistorySearch:
    def test_search_prints_matched_snippets(self, capsys):
        paths = make_meeting("the quarterly budget review")
        assert cli.main(["--history-search", "budget review"]) == 0
        out = capsys.readouterr().out
        assert paths.meeting_id in out
        assert "the quarterly budget review" in out

    def test_search_with_no_matches(self, capsys):
        make_meeting("unrelated content")
        assert cli.main(["--history-search", "missing-token"]) == 0
        assert "No matches" in capsys.readouterr().out

    def test_search_rejects_an_empty_query(self, capsys):
        make_meeting("content")
        assert cli.main(["--history-search", ""]) == 2

    def test_search_rejects_a_bad_date(self, capsys):
        assert cli.main(["--history-search", "x", "--history-after", "09/2026"]) == 2
        assert "YYYY-MM-DD" in capsys.readouterr().out

    def test_search_honors_date_filters(self, capsys):
        make_meeting("budget kickoff")  # today
        assert cli.main(["--history-search", "budget", "--history-after", "2999-01-01"]) == 0
        assert "No matches" in capsys.readouterr().out


class TestHistoryDelete:
    def test_refuses_without_confirm(self, capsys):
        paths = make_meeting()
        assert cli.main(["--history-delete", paths.meeting_id]) == 2
        assert "--confirm" in capsys.readouterr().out
        assert os.path.isdir(paths.meeting_dir)  # nothing was deleted

    def test_deletes_with_confirm(self, capsys):
        paths = make_meeting()
        assert cli.main(["--history-delete", paths.meeting_id, "--confirm"]) == 0
        assert not os.path.isdir(paths.meeting_dir)
        assert "Deleted" in capsys.readouterr().out

    def test_delete_all_with_confirm(self, capsys):
        make_meeting("one")
        make_meeting("two")
        assert cli.main(["--history-delete", "all", "--confirm"]) == 0
        root = ms.meeting_root(FakeCfg())
        assert os.path.isdir(root) and os.listdir(root) == []

    def test_unknown_meeting_is_exit_one(self, capsys):
        make_meeting()
        assert cli.main(["--history-delete", "no-such-meeting", "--confirm"]) == 1
        assert "No such stored meeting" in capsys.readouterr().out


class TestHistoryExport:
    def test_refuses_without_a_destination(self, capsys):
        paths = make_meeting()
        assert cli.main(["--history-export", paths.meeting_id]) == 2
        assert "destination" in capsys.readouterr().out

    def test_exports_to_the_given_directory(self, capsys, tmp_path):
        paths = make_meeting("exportable")
        dest = str(tmp_path / "out")
        assert cli.main(["--history-export", paths.meeting_id, dest]) == 0
        out = capsys.readouterr().out
        assert paths.meeting_id in out
        assert os.path.isdir(os.path.join(dest, paths.meeting_id))
        assert os.path.exists(paths.transcript_path)  # corpus copy untouched

    def test_unknown_meeting_is_exit_one(self, capsys, tmp_path):
        make_meeting()
        assert cli.main(["--history-export", "no-such-meeting", str(tmp_path / "d")]) == 1
