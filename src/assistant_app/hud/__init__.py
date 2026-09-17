"""Menu-bar HUD package (W2).

The AppKit-dependent pieces (``menubar``, ``settings_window``) lazily import
PyObjC inside their classes so this package — and the pure-logic state machine
in ``state`` — stays importable on Linux CI (pre-flight R7).
"""
