"""Menu-bar HUD package (W2).

Pure-logic surfaces (``state``, ``actions``) import anywhere — Linux CI tests
them directly. The AppKit-dependent module (``menubar``) raises ImportError
off-darwin; :func:`create_menu_bar_hud` is the platform-aware entry point that
encapsulates that check (pre-flight R7: explicit degradation, never silent).
"""

from __future__ import annotations

import sys

from assistant_app.hud.state import HUDSnapshot, HUDStateMachine
from assistant_app.utils.logging_config import get_logger

logger = get_logger(__name__)


def create_menu_bar_hud(state_machine: HUDStateMachine, actions) -> object | None:
    """Create the NSStatusItem HUD on darwin; return None (loudly) elsewhere.

    Callers must handle ``None`` — e.g. fall back to the CLI indicator — so a
    non-macOS environment degrades with a logged warning instead of crashing.
    """
    if sys.platform != "darwin":
        logger.warning("Menu-bar HUD unavailable on %s; falling back to CLI indicator", sys.platform)
        return None
    from assistant_app.hud.menubar import MenuBarHUD

    return MenuBarHUD.alloc().initWithStateMachine_actions_(state_machine, actions)


__all__ = ["HUDSnapshot", "HUDStateMachine", "create_menu_bar_hud"]
