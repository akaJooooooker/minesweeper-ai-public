"""Click-through-safe full-screen overlay for selecting a visible grid."""

from __future__ import annotations

import ctypes
import tkinter as tk
from typing import Callable

from .windows_coordinates import enable_dpi_awareness, physical_cursor_position


SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79

class GridCalibrationOverlay:
    """Capture two screen positions without forwarding either click to the game."""

    def __init__(
        self,
        root: tk.Tk,
        *,
        on_complete: Callable[[tuple[int, int], tuple[int, int]], None],
        on_cancel: Callable[[], None],
    ) -> None:
        self.on_complete = on_complete
        self.on_cancel = on_cancel
        self.first: tuple[int, int] | None = None
        user32 = ctypes.windll.user32
        self.left = int(user32.GetSystemMetrics(SM_XVIRTUALSCREEN))
        self.top = int(user32.GetSystemMetrics(SM_YVIRTUALSCREEN))
        self.width = int(user32.GetSystemMetrics(SM_CXVIRTUALSCREEN))
        self.height = int(user32.GetSystemMetrics(SM_CYVIRTUALSCREEN))

        self.window = tk.Toplevel(root)
        self.window.overrideredirect(True)
        self.window.attributes("-topmost", True)
        self.window.attributes("-alpha", 0.28)
        self.window.geometry(
            f"{self.width}x{self.height}{self.left:+d}{self.top:+d}"
        )
        self.window.configure(cursor="crosshair")
        self.canvas = tk.Canvas(
            self.window,
            background="#000000",
            highlightthickness=0,
            cursor="crosshair",
        )
        self.canvas.pack(fill="both", expand=True)
        self.prompt = self.canvas.create_text(
            self.width // 2,
            70,
            text="1 / 2：直接點一下棋盤『左上第一格』的正中央　（Esc 取消）",
            fill="#ffffff",
            font=("Microsoft JhengHei UI", 22, "bold"),
        )
        self.window.bind("<Button-1>", self._click)
        self.window.bind("<Escape>", self._cancel)
        self.window.grab_set()
        self.window.focus_force()

    def _click(self, event: tk.Event) -> None:
        # Tk coordinates can be DPI-virtualised while Pillow captures physical pixels.
        position = physical_cursor_position()
        if self.first is None:
            self.first = position
            self.canvas.create_oval(
                int(event.x) - 12,
                int(event.y) - 12,
                int(event.x) + 12,
                int(event.y) + 12,
                outline="#66dd66",
                width=4,
            )
            self.canvas.itemconfigure(
                self.prompt,
                text="2 / 2：再點一下棋盤『右下最後一格』的正中央　（Esc 取消）",
            )
            return
        second = position
        first = self.first
        self._destroy()
        self.on_complete(first, second)

    def _cancel(self, _event: tk.Event | None = None) -> None:
        self._destroy()
        self.on_cancel()

    def _destroy(self) -> None:
        try:
            self.window.grab_release()
        except tk.TclError:
            pass
        self.window.destroy()
