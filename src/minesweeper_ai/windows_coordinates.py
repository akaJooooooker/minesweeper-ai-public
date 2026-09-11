"""DPI-stable Windows cursor coordinates used by desktop capture and clicking."""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import sys


def enable_dpi_awareness() -> None:
    """Prefer real-pixel coordinates for every subsequently created window."""
    if sys.platform != "win32":
        return
    user32 = ctypes.windll.user32
    try:
        if user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            return
    except (AttributeError, OSError):
        pass
    try:
        user32.SetProcessDPIAware()
    except (AttributeError, OSError):
        pass


def physical_cursor_position() -> tuple[int, int]:
    """Return the cursor in physical desktop pixels, bypassing DPI virtualisation."""
    if sys.platform != "win32":
        raise RuntimeError("desktop cursor access is currently Windows-only")
    user32 = ctypes.windll.user32
    point = ctypes.wintypes.POINT()
    getter = getattr(user32, "GetPhysicalCursorPos", None)
    if getter is not None and getter(ctypes.byref(point)):
        return int(point.x), int(point.y)
    if not user32.GetCursorPos(ctypes.byref(point)):
        raise OSError("GetPhysicalCursorPos/GetCursorPos failed")
    return int(point.x), int(point.y)


def set_physical_cursor_position(coord: tuple[int, int]) -> None:
    """Move the cursor using the same physical-pixel system as Pillow capture."""
    if sys.platform != "win32":
        raise RuntimeError("desktop cursor access is currently Windows-only")
    user32 = ctypes.windll.user32
    x, y = map(int, coord)
    setter = getattr(user32, "SetPhysicalCursorPos", None)
    if setter is not None and setter(x, y):
        return
    if not user32.SetCursorPos(x, y):
        raise OSError("SetPhysicalCursorPos/SetCursorPos failed")
