"""Escaping helpers for interpolating untrusted text into scripting hosts.

``applescript_escape`` makes arbitrary user/LLM text safe to embed inside an
AppleScript string literal. The previous type_text implementation escaped only
backslash and double-quote, so newlines, tabs, and other control characters in
LLM-generated text could terminate or corrupt the AppleScript and send
keystrokes somewhere unintended.
"""

_APPLESCRIPT_ESCAPES: tuple[tuple[str, str], ...] = (
    ("\\", "\\\\"),  # backslash first — later escapes introduce backslashes
    ('"', '\\"'),
    ("\n", "\\n"),
    ("\r", "\\r"),
    ("\t", "\\t"),
)


def applescript_escape(text: str) -> str:
    """Escape ``text`` for safe interpolation into an AppleScript string literal.

    Backslashes, quotes, newlines, carriage returns, and tabs become AppleScript
    escape sequences; any other control characters are stripped because the
    keystroke pathway cannot type them meaningfully.
    """
    escaped = text
    for raw, literal in _APPLESCRIPT_ESCAPES:
        escaped = escaped.replace(raw, literal)
    return "".join(ch for ch in escaped if ord(ch) >= 32)
