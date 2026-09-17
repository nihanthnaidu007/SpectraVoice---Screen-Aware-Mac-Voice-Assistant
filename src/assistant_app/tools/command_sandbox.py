"""Sandboxed command execution for the run_command tool.

Threat model: the command string is chosen by the LLM. The previous defense
(prefix match against a whitelist + ``shell=True``) was bypassable — any
payload *starting with* a whitelisted prefix executed, so ``ls; curl … | sh``
and ``python -c '…'`` ran arbitrary code.

This module enforces instead:

1. **No shell** — callers execute the returned argv list with ``shell=False``;
   separators, redirects, and command substitution are structurally inert.
2. **Strict binary allowlist** — the first token must be a bare name present
   in ``ALLOWED_BINARIES`` (no path separators, so ``/bin/zsh`` or ``./evil``
   cannot smuggle a binary in); it is resolved to an absolute path via PATH.
3. **Fixed argv patterns** — every argument must match one of the allowed
   styles for that binary (flags, subcommands, safe relative paths, numbers);
   code-execution flags like ``find -exec`` are denied by name.
4. **Metacharacter rejection** — any payload containing shell metacharacters
   (``;``, ``|``, ``&``, backticks, ``$()``, redirects) is rejected outright.
"""

from __future__ import annotations

import re
import shlex
import shutil
from dataclasses import dataclass


class SandboxError(ValueError):
    """Raised when a command is not permitted by the sandbox."""


# Characters with no legitimate use in an allowlisted read-only command.
# Checked against the raw payload before parsing, so even quoted metacharacters
# are refused (fail-closed; no allowlisted usage needs them).
SHELL_METACHARACTERS = (";", "|", "&", "`", "$", "(", ")", "<", ">", "\n", "\r", "\x00")

# Argument styles: each token an allowlisted binary receives must match at
# least one style declared for that binary.
_FLAG_RE = re.compile(r"^-{1,2}[A-Za-z][A-Za-z0-9-]{0,19}$")       # -l, --help
_NUMERIC_RE = re.compile(r"^\d{1,9}$")                              # head -n 20
_PLAIN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._+@%:,/-]{0,127}$")  # words: echo/git/which
_MAX_TOKEN_LEN = 128


@dataclass(frozen=True)
class BinaryRule:
    """What an allowlisted binary may be invoked with."""

    # Positional styles this binary accepts. ("flag" is always permitted and
    # validated separately, so it is not part of this tuple.)
    arg_styles: tuple[str, ...]
    max_args: int
    # Flags denied for this binary even though they match the flag pattern.
    denied_flags: frozenset[str] = frozenset()
    # For the "subcommand" style: exact words allowed as first positional.
    allowed_subcommands: frozenset[str] = frozenset()


# Code execution, package installation, network fetch, and process control
# binaries are deliberately absent: python, osascript, curl, wget, open, pip,
# npm, node, brew, say, screencapture, networksetup, diskutil, defaults.
ALLOWED_BINARIES: dict[str, BinaryRule] = {
    # Directory / system information
    "pwd": BinaryRule(arg_styles=(), max_args=0),
    "date": BinaryRule(arg_styles=("flag",), max_args=2),
    "whoami": BinaryRule(arg_styles=(), max_args=0),
    "hostname": BinaryRule(arg_styles=(), max_args=0),
    "uptime": BinaryRule(arg_styles=(), max_args=0),
    "uname": BinaryRule(arg_styles=("flag",), max_args=2),
    "sw_vers": BinaryRule(arg_styles=("flag",), max_args=2),
    "df": BinaryRule(arg_styles=("flag",), max_args=2),
    "ps": BinaryRule(arg_styles=("flag", "plain"), max_args=4),
    "top": BinaryRule(arg_styles=("flag", "numeric"), max_args=4),
    "ls": BinaryRule(arg_styles=("flag", "path"), max_args=8),
    "du": BinaryRule(arg_styles=("flag", "path"), max_args=4),
    "wc": BinaryRule(arg_styles=("flag", "path"), max_args=4),
    "sort": BinaryRule(arg_styles=("flag", "numeric", "path"), max_args=4),
    "uniq": BinaryRule(arg_styles=("flag", "path"), max_args=4),
    # File content readers — relative, non-hidden paths only (no /etc reads,
    # no traversal into ~/.ssh)
    "cat": BinaryRule(arg_styles=("flag", "path"), max_args=4),
    "head": BinaryRule(arg_styles=("flag", "numeric", "path"), max_args=4),
    "tail": BinaryRule(arg_styles=("flag", "numeric", "path"), max_args=4),
    "grep": BinaryRule(arg_styles=("flag", "numeric", "path"), max_args=6),
    "find": BinaryRule(
        arg_styles=("flag", "path"),
        max_args=8,
        denied_flags=frozenset(
            {"-exec", "-execdir", "-ok", "-okdir", "-delete", "-f", "-fprintf"}
        ),
    ),
    # Print-only / lookup utilities
    "echo": BinaryRule(arg_styles=("plain",), max_args=16),
    "which": BinaryRule(arg_styles=("plain",), max_args=3),
    "where": BinaryRule(arg_styles=("plain",), max_args=3),
    # VCS inspection only
    "git": BinaryRule(
        arg_styles=("subcommand", "flag"),
        max_args=4,
        allowed_subcommands=frozenset({"status", "log", "branch", "diff"}),
    ),
}


def resolve_binary(name: str) -> str | None:
    """Resolve an allowlisted bare binary name to its absolute path.

    Returns None for anything not allowlisted or not installed — including
    names containing path separators, so a payload can never select its own
    executable.
    """
    if name not in ALLOWED_BINARIES:
        return None
    if "/" in name or "\\" in name:
        return None
    return shutil.which(name)


def _is_safe_path(token: str) -> bool:
    """Relative, non-hidden, non-traversing path fragment (no separators)."""
    if not token or len(token) > _MAX_TOKEN_LEN:
        return False
    if token.startswith("-"):
        return False
    if token != "." and token.startswith("."):
        return False  # hidden files/dirs (.ssh, .env, ...)
    if ".." in token:
        return False
    if "/" in token or "\\" in token:
        return False  # absolute paths and traversal are not allowed
    return all(ch.isprintable() for ch in token)


def _check_token(binary: str, rule: BinaryRule, token: str) -> None:
    """Validate one argument token against the binary's rule; raise on failure."""
    if len(token) > _MAX_TOKEN_LEN:
        raise SandboxError(f"rejected: argument too long for '{binary}': {token[:40]!r}")

    if token.startswith("-"):
        if not _FLAG_RE.match(token):
            raise SandboxError(
                f"rejected: argument {token[:40]!r} is not an accepted flag for '{binary}'"
            )
        if token in rule.denied_flags:
            raise SandboxError(
                f"rejected: flag '{token}' is denied for '{binary}' (code execution vector)"
            )
        return

    styles = rule.arg_styles
    if "subcommand" in styles and token in rule.allowed_subcommands:
        return
    if "numeric" in styles and _NUMERIC_RE.match(token):
        return
    if "plain" in styles and _PLAIN_RE.match(token):
        return
    if "path" in styles and _is_safe_path(token):
        return

    raise SandboxError(
        f"rejected: argument {token[:40]!r} does not match any allowed pattern for '{binary}'"
    )


def build_sandbox_argv(command: str) -> list[str]:
    """Translate a command string into a sandboxed argv list.

    Raises:
        SandboxError: If the payload contains shell metacharacters, cannot be
            parsed, or names/argues a binary outside the allowlist.
    """
    if not command or not command.strip():
        raise SandboxError("rejected: empty command")

    for ch in SHELL_METACHARACTERS:
        if ch in command:
            raise SandboxError(
                f"rejected: command contains shell metacharacter {ch!r}"
            )

    try:
        tokens = shlex.split(command)
    except ValueError as e:
        raise SandboxError(f"rejected: could not parse command ({e})") from e

    if not tokens:
        raise SandboxError("rejected: empty command")

    name, *args = tokens
    resolved = resolve_binary(name)
    if resolved is None:
        raise SandboxError(f"rejected: '{name}' is not in the sandbox allowlist")

    rule = ALLOWED_BINARIES[name]
    if len(args) > rule.max_args:
        raise SandboxError(
            f"rejected: too many arguments for '{name}' (max {rule.max_args})"
        )

    for token in args:
        _check_token(name, rule, token)

    return [resolved, *args]
