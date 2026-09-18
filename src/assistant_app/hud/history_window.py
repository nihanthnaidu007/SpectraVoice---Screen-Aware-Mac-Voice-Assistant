"""History / privacy dashboard window (W4 D2) — AppKit, darwin-only.

Renders the pure history model (:mod:`assistant_app.hud.history_model`):
consent state, retention posture, and every stored meeting with per-meeting
actions (view summary, export, delete) plus a refresh and a confirmation-
gated delete-all.

Threading contract (the sharpest W4 edge, H1/R1): the HUD owns the main
thread inside NSApplication.run() — a corpus scan here would freeze icon
updates and the recording-red-dot path. So the window OPENS instantly with a
placeholder, and every scan/file op runs on a worker thread; finished values
are marshaled back with ``performSelectorOnMainThread_`` (the menu-bar
snapshot precedent). Confirmations (NSAlert / checkbox / NSOpenPanel) happen
on the main thread because they are momentary UI, never corpus IO.

Destructive actions are never a single click: Delete asks for confirmation,
Delete All needs BOTH a checkbox and a confirmation dialog; the ops layer
additionally refuses unconfirmed calls (structural gate in history_ops).

Importing this module on a non-darwin platform raises ImportError with a
clear message (tests assert this explicitly — never a silent skip).
"""

from __future__ import annotations

import threading
from typing import Any

from assistant_app.hud.history_model import (
    HistorySnapshot,
    format_history_snapshot,
    format_meeting_line,
)
from assistant_app.services.history_qa import format_answer
from assistant_app.utils.logging_config import get_logger

logger = get_logger(__name__)

try:
    from AppKit import (
        NSAlert,
        NSAlertFirstButtonReturn,
        NSBackingStoreBuffered,
        NSButton,
        NSEdgeInsetsMake,
        NSFont,
        NSOpenPanel,
        NSScrollView,
        NSStackView,
        NSTextField,
        NSTextView,
        NSUserInterfaceLayoutOrientationHorizontal,
        NSUserInterfaceLayoutOrientationVertical,
        NSViewWidthSizable,
        NSWindow,
        NSWindowStyleMaskClosable,
        NSWindowStyleMaskMiniaturizable,
        NSWindowStyleMaskResizable,
        NSWindowStyleMaskTitled,
    )
    from Foundation import NSMakeRect, NSObject
except ImportError as exc:  # pragma: no cover - exercised on non-darwin only
    raise ImportError(
        "assistant_app.hud.history_window is darwin-only: AppKit/PyObjC is "
        "unavailable on this platform. The pure history model "
        "(assistant_app.hud.history_model) has no such dependency."
    ) from exc

_WINDOW_W, _WINDOW_H = 760, 560
_DETAILS_H = 180.0

# Corpora are small (scan-on-query, locked decision), but the window must not
# grow unbounded either — past this, point at the CLI for the tail.
MAX_RENDERED_MEETINGS = 100

# The provider duck-types SpectraVoiceAssistant: config_manager (for the
# history.export_dir prefill) plus the thin W4/W1 adapter methods:
#   history_snapshot() -> HistorySnapshot        (worker-thread caller)
#   history_summary_text(meeting_id) -> str      (worker-thread caller)
#   history_export_meeting(meeting_id, dest) -> str
#   history_delete_meeting(meeting_id) -> None   (raises on failure)
#   history_delete_all() -> int                  (raises on failure)
#   history_ask(question) -> HistoryAnswer       (worker-thread caller; W1)


class _HistoryController(NSObject):
    """Window controller: renders snapshots; all corpus IO off-thread."""

    def initWithProvider_(self, provider):
        # PyObjC two-phase init: super().init() may return a different
        # instance (or None), so configure and return that object directly.
        controller = super().init()
        if controller is None:
            return None
        controller._provider = provider
        controller._snapshot: HistorySnapshot | None = None
        controller._details = None
        controller._question = None
        controller._delete_all_box = None
        controller._delete_all_button = None
        controller.window = None
        return controller

    # --- UI construction (main thread) ---------------------------------

    def build_window(self) -> None:
        scroll = NSScrollView.alloc().initWithFrame_(NSMakeRect(0, 0, _WINDOW_W, _WINDOW_H))
        scroll.setHasVerticalScroller_(True)
        self._scroll = scroll

        window = NSWindow.alloc().initWithContentRect_styleMask_title_backing_defer_(
            NSMakeRect(0, 0, _WINDOW_W, _WINDOW_H),
            NSWindowStyleMaskTitled
            | NSWindowStyleMaskClosable
            | NSWindowStyleMaskMiniaturizable
            | NSWindowStyleMaskResizable,
            "SpectraVoice History",
            NSBackingStoreBuffered,
            False,
        )
        window.setContentView_(scroll)
        window.setReleasedWhenClosed_(False)
        window.setDelegate_(self)
        window.center()
        self.window = window

        # First render is a placeholder; the scan runs off-thread and the
        # finished snapshot arrives via applySnapshot_ (H1: no IO on main).
        self._render_placeholder()
        self.refresh_(None)

    # --- rendering (main thread) ----------------------------------------

    def _render_placeholder(self) -> None:
        self._set_document_text("History loads in a moment — scanning the local corpus off-thread…")

    def _set_document_text(self, text: str, meetings: tuple[dict, ...] = ()) -> None:
        """Rebuild the document view: ask row + header + one row per meeting."""
        content = NSStackView.alloc().init()
        content.setOrientation_(NSUserInterfaceLayoutOrientationVertical)
        content.setSpacing_(4)
        content.setEdgeInsets_(NSEdgeInsetsMake(16, 16, 16, 16))

        # W1 D2: the question row is part of EVERY rebuild so it survives
        # snapshot refreshes; typed text is preserved across rebuilds.
        question_text = self._question.stringValue() if self._question is not None else ""
        content.addArrangedSubview_(self._ask_row(question_text or ""))

        header = NSTextField.labelWithString_(text)
        header.setFont_(NSFont.systemFontOfSize_(11))
        content.addArrangedSubview_(header)

        for index, meeting in enumerate(meetings[:MAX_RENDERED_MEETINGS]):
            content.addArrangedSubview_(self._meeting_row(index, meeting))
        hidden = len(meetings) - min(len(meetings), MAX_RENDERED_MEETINGS)
        if hidden > 0:
            more = NSTextField.labelWithString_(f"… and {hidden} more — use the CLI (--history-list)")
            more.setFont_(NSFont.systemFontOfSize_(10))
            content.addArrangedSubview_(more)

        content.addArrangedSubview_(self._footer_row())

        details_scroll = NSScrollView.alloc().initWithFrame_(NSMakeRect(0, 0, _WINDOW_W, _DETAILS_H))
        details = NSTextView.alloc().initWithFrame_(NSMakeRect(0, 0, _WINDOW_W, _DETAILS_H))
        details.setEditable_(False)
        details.setSelectable_(True)
        self._details = details
        details_scroll.setDocumentView_(details)
        details_scroll.setHasVerticalScroller_(True)
        content.addArrangedSubview_(details_scroll)

        content.setAutoresizingMask_(NSViewWidthSizable)
        self._scroll.setDocumentView_(content)

    def applySnapshot_(self, snapshot: HistorySnapshot) -> None:
        """Render a finished snapshot (main thread, from the worker marshal)."""
        self._snapshot = snapshot
        header = format_history_snapshot(
            HistorySnapshot(meetings=(), retention=snapshot.retention, consent=snapshot.consent)
        )
        self._set_document_text(header, snapshot.meetings)

    def applyDetails_(self, message: Any) -> None:
        """Render a worker outcome (summary text, export/delete result)."""
        if self._details is not None:
            self._details.setText_(str(message))

    # --- worker dispatch (off-thread corpus IO) --------------------------

    def refresh_(self, sender) -> None:
        def work() -> None:
            try:
                snapshot = self._provider.history_snapshot()
                self.performSelectorOnMainThread_withObject_waitUntilDone_("applySnapshot:", snapshot, False)
            except Exception as exc:  # surfaced in the window, never silent
                logger.exception("History snapshot failed")
                self.performSelectorOnMainThread_withObject_waitUntilDone_(
                    "applyDetails:", f"Snapshot failed: {exc}", False
                )

        threading.Thread(target=work, name="sv-history-scan", daemon=True).start()

    def askAction_(self, sender) -> None:
        """The W1 ask: dispatch the question off-thread (H1 — no LLM/scan IO
        on the main thread); the finished answer renders via applyDetails_."""
        question = ""
        if self._question is not None:
            question = (self._question.stringValue() or "").strip()
        if not question:
            self.applyDetails_("Type a question about your stored meetings first.")
            return
        provider = self._provider
        self.applyDetails_("Searching your stored meetings…")

        def work() -> None:
            try:
                answer = provider.history_ask(question)
                self.performSelectorOnMainThread_withObject_waitUntilDone_(
                    "applyDetails:", format_answer(answer), False
                )
            except Exception as exc:  # surfaced in the window, never silent
                logger.exception("History Q&A failed")
                self.performSelectorOnMainThread_withObject_waitUntilDone_("applyDetails:", f"Ask failed: {exc}", False)

        threading.Thread(target=work, name="sv-history-ask", daemon=True).start()

    def viewSummaryAction_(self, sender) -> None:
        meeting_id = self._meeting_id(sender)
        if meeting_id is None:
            return
        provider = self._provider

        def work() -> None:
            try:
                summary = provider.history_summary_text(meeting_id)
                text = summary or f"No summary stored for {meeting_id}."
                self.performSelectorOnMainThread_withObject_waitUntilDone_("applyDetails:", text, False)
            except Exception as exc:
                logger.exception("History summary read failed")
                self.performSelectorOnMainThread_withObject_waitUntilDone_(
                    "applyDetails:", f"Summary read failed: {exc}", False
                )

        threading.Thread(target=work, name="sv-history-summary", daemon=True).start()

    def exportAction_(self, sender) -> None:
        meeting_id = self._meeting_id(sender)
        if meeting_id is None:
            return
        dest = self._choose_export_destination()
        if not dest:
            return
        provider = self._provider

        def work() -> None:
            try:
                out = provider.history_export_meeting(meeting_id, dest)
                self.performSelectorOnMainThread_withObject_waitUntilDone_(
                    "applyDetails:", f"Exported {meeting_id} → {out}", False
                )
            except Exception as exc:
                logger.exception("History export failed")
                self.performSelectorOnMainThread_withObject_waitUntilDone_(
                    "applyDetails:", f"Export failed: {exc}", False
                )

        threading.Thread(target=work, name="sv-history-export", daemon=True).start()

    def deleteAction_(self, sender) -> None:
        meeting_id = self._meeting_id(sender)
        if meeting_id is None:
            return
        if not self._confirm(
            "Delete Meeting",
            f"Permanently delete meeting {meeting_id}?\n"
            "Its transcript, summary, and audio clips will be removed from disk.",
        ):
            return
        provider = self._provider

        def work() -> None:
            try:
                provider.history_delete_meeting(meeting_id)
                self.performSelectorOnMainThread_withObject_waitUntilDone_(
                    "applyDetails:", f"Deleted meeting {meeting_id}.", False
                )
                self.performSelectorOnMainThread_withObject_waitUntilDone_("refresh:", None, False)
            except Exception as exc:
                logger.exception("History delete failed")
                self.performSelectorOnMainThread_withObject_waitUntilDone_(
                    "applyDetails:", f"Delete failed: {exc}", False
                )

        threading.Thread(target=work, name="sv-history-delete", daemon=True).start()

    def deleteAllBoxAction_(self, sender) -> None:
        # Two-key confirm: the button stays disabled until the box is ticked.
        if self._delete_all_button is not None and self._delete_all_box is not None:
            self._delete_all_button.setEnabled_(bool(self._delete_all_box.state()))

    def deleteAllAction_(self, sender) -> None:
        if not self._confirm(
            "Delete ALL Meetings",
            "Permanently delete EVERY stored meeting (transcripts, summaries, clips)? This cannot be undone.",
        ):
            return
        provider = self._provider

        def work() -> None:
            try:
                removed = provider.history_delete_all()
                self.performSelectorOnMainThread_withObject_waitUntilDone_(
                    "applyDetails:", f"Deleted {removed} meeting(s).", False
                )
                self.performSelectorOnMainThread_withObject_waitUntilDone_("refresh:", None, False)
            except Exception as exc:
                logger.exception("History delete-all failed")
                self.performSelectorOnMainThread_withObject_waitUntilDone_(
                    "applyDetails:", f"Delete-all failed: {exc}", False
                )

        threading.Thread(target=work, name="sv-history-nuke", daemon=True).start()

    # --- helpers ---------------------------------------------------------

    def _ask_row(self, preserve_text: str = "") -> Any:
        """The W1 question-input row: text field + Ask button (askAction_)."""
        row = NSStackView.alloc().init()
        row.setOrientation_(NSUserInterfaceLayoutOrientationHorizontal)
        row.setSpacing_(8)
        self._question = NSTextField.alloc().initWithFrame_(NSMakeRect(0, 0, 560.0, 24.0))
        self._question.setEditable_(True)
        self._question.setSelectable_(True)
        self._question.setPlaceholderString_("Ask your stored meetings…")
        self._question.setStringValue_(preserve_text)
        self._question.setTarget_(self)
        self._question.setAction_("askAction:")  # Return in the field asks
        self._question.widthAnchor().constraintEqualToConstant_(560.0).setActive_(True)
        row.addArrangedSubview_(self._question)
        row.addArrangedSubview_(NSButton.buttonWithTitle_target_action_("Ask", self, "askAction:"))
        return row

    def _focus_question(self) -> None:
        """Keyboard focus on the question field (the Ask History entry path)."""
        if self._question is not None and self.window is not None:
            self.window.makeKeyAndOrderFront_(None)
            self.window.makeFirstResponder_(self._question)

    def _meeting_row(self, index: int, meeting: dict) -> Any:
        row = NSStackView.alloc().init()
        row.setOrientation_(NSUserInterfaceLayoutOrientationHorizontal)
        row.setSpacing_(8)

        label = NSTextField.labelWithString_(format_meeting_line(meeting))
        label.widthAnchor().constraintEqualToConstant_(440.0).setActive_(True)
        row.addArrangedSubview_(label)

        row.addArrangedSubview_(self._tagged_button("View Summary", "viewSummaryAction:", index, 120.0))
        row.addArrangedSubview_(self._tagged_button("Export…", "exportAction:", index, 80.0))
        row.addArrangedSubview_(self._tagged_button("Delete…", "deleteAction:", index, 70.0))
        return row

    def _tagged_button(self, title: str, action: str, index: int, width: float) -> Any:
        button = NSButton.buttonWithTitle_target_action_(title, self, action)
        button.setTag_(index)
        button.widthAnchor().constraintEqualToConstant_(width).setActive_(True)
        return button

    def _footer_row(self) -> Any:
        row = NSStackView.alloc().init()
        row.setOrientation_(NSUserInterfaceLayoutOrientationHorizontal)
        row.setSpacing_(12)
        refresh_btn = NSButton.buttonWithTitle_target_action_("Refresh", self, "refresh:")
        row.addArrangedSubview_(refresh_btn)
        self._delete_all_box = NSButton.checkboxWithTitle_target_action_(
            "I understand Delete All is permanent", self, "deleteAllBoxAction:"
        )
        row.addArrangedSubview_(self._delete_all_box)
        self._delete_all_button = NSButton.buttonWithTitle_target_action_("Delete All…", self, "deleteAllAction:")
        self._delete_all_button.setEnabled_(False)  # checkbox gates it (two-key confirm)
        row.addArrangedSubview_(self._delete_all_button)
        return row

    def _meeting_id(self, sender) -> str | None:
        meetings = self._snapshot.meetings if self._snapshot is not None else ()
        index = int(sender.tag())
        if 0 <= index < len(meetings):
            return meetings[index]["meeting_id"]
        return None

    def _choose_export_destination(self) -> str | None:
        """Directory picker (momentary, main thread — no corpus IO inside)."""
        panel = NSOpenPanel.alloc().init()
        panel.setCanChooseDirectories_(True)
        panel.setCanChooseFiles_(False)
        panel.setCanCreateDirectories_(True)
        panel.setPrompt_("Export Here")
        try:
            export_dir = self._provider.config_manager.config.history.export_dir or ""
        except Exception:  # provider without config — ask without prefill
            export_dir = ""
        if export_dir:
            from Foundation import NSURL  # local import: only needed on this path

            panel.setDirectoryURL_(NSURL.fileURLWithPath_(export_dir))
        if int(panel.runModal()) != NSAlertFirstButtonReturn:
            return None
        return panel.URL().path()

    def _confirm(self, title: str, message: str) -> bool:
        alert = NSAlert.alloc().init()
        alert.setMessageText_(title)
        alert.setInformativeText_(message)
        alert.addButtonWithTitle_("Delete")
        alert.addButtonWithTitle_("Cancel")
        return int(alert.runModal()) == NSAlertFirstButtonReturn

    def windowWillClose_(self, notification) -> None:  # NSWindow delegate
        _release_controller(self)


# Module-level reference keeps the open window/controller alive.
_HISTORY_CONTROLLER: _HistoryController | None = None


def _release_controller(controller: _HistoryController) -> None:
    global _HISTORY_CONTROLLER
    if _HISTORY_CONTROLLER is controller:
        _HISTORY_CONTROLLER = None


def open_history_window(provider, *, focus_question: bool = False) -> None:
    """Open (or focus) the history / privacy dashboard window.

    ``focus_question`` (W1 D2, the Ask History menu path) additionally puts
    keyboard focus in the question field. Must be called on the main thread —
    the HUD menu is the entry point, so the NSApplication run loop is already
    live. The first snapshot scan runs on a worker thread; the window shows a
    placeholder until it lands (H1).
    """
    global _HISTORY_CONTROLLER
    if _HISTORY_CONTROLLER is not None and _HISTORY_CONTROLLER.window is not None:
        _HISTORY_CONTROLLER.window.makeKeyAndOrderFront_(None)
        if focus_question:
            _HISTORY_CONTROLLER._focus_question()
        return
    controller = _HistoryController.alloc().initWithProvider_(provider)
    controller.build_window()
    controller.window.makeKeyAndOrderFront_(None)
    if focus_question:
        controller._focus_question()
    _HISTORY_CONTROLLER = controller
