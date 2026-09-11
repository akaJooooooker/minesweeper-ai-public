"""Windows desktop front-end for watching V44 play a visible board."""

from __future__ import annotations

# Run this before importing Tk/Pillow or creating UI objects. Physical cursor
# APIs remain the safeguard if Windows fixed the process DPI mode already.
from .windows_coordinates import enable_dpi_awareness

enable_dpi_awareness()

from dataclasses import dataclass
import os
import ctypes
import queue
import subprocess
import sys
from pathlib import Path
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

from .desktop_live import LiveAction, LiveDecisionEngine, WindowsMouse, detect_region_reset
from .desktop_vision import BoardRead, GridGeometry, HdSkinRecognizer, format_observation
from .screen_calibration import GridCalibrationOverlay
from .game import FLAGGED, UNKNOWN, MinesweeperGame
from .metrics import calculate_3bv, index_of_efficiency, three_bv_per_second
from .replay_deployment import load_replay_deployment


MODEL_CONFIG = Path(__file__).resolve().parents[2] / "models" / "replay-agent-v44.json"
VK_F8 = 0x77
VK_F9 = 0x78
MAX_VISUAL_RETRIES = 10
MAX_UNCHANGED_RETRIES = 15
MAX_CONTRADICTION_RETRIES = 2
MAX_TERMINAL_CONFIRMATIONS = 2
CPU_THREADS = max(2, min(8, os.cpu_count() or 2))
WEB_UPDATE_POLL_SECONDS = 0.025
WIN_SETTLE_SECONDS = 1.2
HOVER_SETTLE_SECONDS = 0.015


@dataclass(frozen=True)
class RunSettings:
    geometry: GridGeometry
    mines: int
    mode: str
    theme: str
    tolerance: int


class DesktopBotApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("AI 踩地雷 V44 — 桌面觀察器")
        self.root.geometry("940x720")
        self.root.minsize(780, 600)
        self.mouse = WindowsMouse()
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.first_cell: tuple[int, int] | None = None
        self.calibration_overlay: GridCalibrationOverlay | None = None
        self.last_cell: tuple[int, int] | None = None
        self.geometry: GridGeometry | None = None
        self._key_down = {VK_F8: False, VK_F9: False}
        self.auto_hidden = False

        self.rows = tk.IntVar(value=16)
        self.cols = tk.IntVar(value=30)
        self.mines = tk.IntVar(value=99)
        self.mode = tk.StringVar(value="standard")
        self.theme = tk.StringVar(value="dark")
        self.tolerance = tk.IntVar(value=58)
        self.delay_ms = tk.IntVar(value=250)
        self.current_delay_ms = 250.0
        self.status = tk.StringVar(value="尚未校準")
        self.calibration = tk.StringVar(value="左上：—　右下：—")
        self.metrics = tk.StringVar(value="點擊 0｜時間 0.00s｜3BV/s —｜網站 IOE —｜重置 0")

        self._build_ui()
        self.root.after(60, self._poll_events)
        self.root.after(80, self._poll_hotkeys)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)

        settings = ttk.LabelFrame(outer, text="棋盤與辨識", padding=10)
        settings.pack(fill="x")
        ttk.Label(settings, text="列").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(settings, from_=2, to=100, textvariable=self.rows, width=6).grid(row=0, column=1)
        ttk.Label(settings, text="欄").grid(row=0, column=2, padx=(12, 0), sticky="w")
        ttk.Spinbox(settings, from_=2, to=100, textvariable=self.cols, width=6).grid(row=0, column=3)
        ttk.Label(settings, text="雷數").grid(row=0, column=4, padx=(12, 0), sticky="w")
        ttk.Spinbox(settings, from_=1, to=999, textvariable=self.mines, width=7).grid(row=0, column=5)
        ttk.Button(settings, text="初級", width=6, command=lambda: self._preset(9, 9, 10)).grid(row=0, column=6, padx=(12, 1))
        ttk.Button(settings, text="中級", width=6, command=lambda: self._preset(16, 16, 40)).grid(row=0, column=7, padx=1)
        ttk.Button(settings, text="高級", width=6, command=lambda: self._preset(16, 30, 99)).grid(row=0, column=8, padx=1)
        ttk.Button(settings, text="地獄", width=6, command=lambda: self._preset(20, 30, 130)).grid(row=0, column=9, padx=1)

        ttk.Label(settings, text="模式").grid(row=1, column=0, pady=(9, 0), sticky="w")
        ttk.Combobox(
            settings,
            textvariable=self.mode,
            values=("standard", "no_guess", "multiplayer"),
            state="readonly",
            width=13,
        ).grid(row=1, column=1, columnspan=2, pady=(9, 0), sticky="w")
        ttk.Label(settings, text="主題").grid(row=1, column=3, pady=(9, 0), sticky="e")
        ttk.Combobox(
            settings,
            textvariable=self.theme,
            values=("dark", "auto", "light"),
            state="readonly",
            width=8,
        ).grid(row=1, column=4, pady=(9, 0), sticky="w")
        ttk.Label(settings, text="顏色容差").grid(row=1, column=5, pady=(9, 0), sticky="e")
        ttk.Spinbox(settings, from_=10, to=120, textvariable=self.tolerance, width=6).grid(row=1, column=6, pady=(9, 0))

        calibration = ttk.LabelFrame(outer, text="畫面校準", padding=10)
        calibration.pack(fill="x", pady=(10, 0))
        ttk.Button(
            calibration,
            text="點擊選取棋盤（左上 → 右下）",
            command=self._begin_click_calibration,
        ).pack(side="left")
        ttk.Label(calibration, textvariable=self.calibration).pack(side="left", padx=8)

        controls = ttk.Frame(outer)
        controls.pack(fill="x", pady=10)
        ttk.Button(controls, text="開啟本機遊戲", command=self._open_local_game).pack(side="left", padx=(0, 8))
        ttk.Label(controls, text="額外點擊延遲").pack(side="left")
        ttk.Scale(
            controls,
            from_=0,
            to=2000,
            variable=self.delay_ms,
            command=self._set_delay,
            orient="horizontal",
            length=240,
        ).pack(side="left", padx=6)
        ttk.Label(controls, textvariable=self.delay_ms, width=5).pack(side="left")
        ttk.Label(controls, text="ms").pack(side="left")
        ttk.Button(controls, text="辨識一次", command=self.inspect_once).pack(side="right")
        ttk.Button(controls, text="停止 F9", command=self.stop).pack(side="right", padx=6)
        ttk.Button(controls, text="開始 F8", command=self.start).pack(side="right")

        ttk.Label(outer, textvariable=self.status, font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Label(outer, textvariable=self.metrics).pack(anchor="w", pady=(3, 8))

        log_frame = ttk.LabelFrame(outer, text="模型看到的盤面與原始決策", padding=6)
        log_frame.pack(fill="both", expand=True)
        self.log = tk.Text(
            log_frame,
            wrap="none",
            font=("Cascadia Mono", 10),
            background="#171717",
            foreground="#e8e8e8",
            insertbackground="white",
        )
        yscroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        xscroll = ttk.Scrollbar(log_frame, orient="horizontal", command=self.log.xview)
        self.log.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.log.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

        ttk.Label(
            outer,
            text="可用「開啟本機遊戲」離線遊玩；本機內建 V44 免校準。觀察器的 F9 為全域停止。",
            foreground="#a44",
        ).pack(anchor="w", pady=(7, 0))

    def _open_local_game(self) -> None:
        subprocess.Popen([sys.executable, "-m", "minesweeper_ai.local_app"],
                         cwd=Path(__file__).resolve().parents[2])

    def _preset(self, rows: int, cols: int, mines: int) -> None:
        self.rows.set(rows)
        self.cols.set(cols)
        self.mines.set(mines)
        self.geometry = None
        self.status.set("尺寸已更改，請重新校準")
    def _set_delay(self, value: str) -> None:
        self.current_delay_ms = float(value)


    def _begin_click_calibration(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showwarning("正在執行", "請先按 F9 停止。")
            return
        self.status.set("校準中：直接點棋盤左上第一格，再點右下最後一格")
        self.root.withdraw()
        self.root.update_idletasks()
        self.calibration_overlay = GridCalibrationOverlay(
            self.root,
            on_complete=self._complete_click_calibration,
            on_cancel=self._cancel_click_calibration,
        )

    def _complete_click_calibration(self, first: tuple[int, int], last: tuple[int, int]) -> None:
        self.calibration_overlay = None
        self.first_cell = first
        self._restore_window()
        self.last_cell = last
        try:
            self.geometry = GridGeometry.from_corner_centres(
                first,
                last,
                rows=self.rows.get(),
                cols=self.cols.get(),
            )
        except ValueError as error:
            self.geometry = None
            self.status.set(f"校準失敗：{error}")
            return
        cell_width = (self.geometry.right - self.geometry.left) / self.geometry.cols
        cell_height = (self.geometry.bottom - self.geometry.top) / self.geometry.rows
        self.calibration.set(
            f"左上：{first}　右下點：{last}　自動校正：{cell_width:.0f}×{cell_height:.0f} 像素"
        )
        self.status.set(
            f"已校準：({self.geometry.left},{self.geometry.top})–"
            f"({self.geometry.right},{self.geometry.bottom})；請先按『辨識一次』確認有看到綠色 X"
        )

    def _cancel_click_calibration(self) -> None:
        self.calibration_overlay = None
        self.status.set("已取消校準")
        self._restore_window()

    def _settings(self) -> RunSettings | None:
        if self.geometry is None:
            messagebox.showerror("尚未校準", "請按『點擊選取棋盤』，再依序點左上格與右下格。")
            return None
        try:
            mines = int(self.mines.get())
            tolerance = int(self.tolerance.get())
        except (ValueError, tk.TclError):
            messagebox.showerror("設定錯誤", "列、欄、雷數與容差必須是整數。")
            return None
        if mines < 1 or mines >= self.geometry.rows * self.geometry.cols:
            messagebox.showerror("設定錯誤", "雷數必須小於總格數。")
            return None
        return RunSettings(
            self.geometry,
            mines,
            self.mode.get(),
            self.theme.get(),
            tolerance,
        )

    def inspect_once(self) -> None:
        settings = self._settings()
        if settings is None:
            return
        self.status.set("暫時隱藏視窗並讀取棋盤…")
        self.root.withdraw()
        self.root.after(180, lambda: self._inspect_hidden(settings))

    def _inspect_hidden(self, settings: RunSettings) -> None:
        try:
            read = HdSkinRecognizer(
                colour_tolerance=settings.tolerance,
                theme=settings.theme,
            ).read_screen(settings.geometry, total_mines=settings.mines)
        except Exception as error:
            self._restore_window()
            messagebox.showerror("辨識失敗", str(error))
            return
        self._restore_window()
        self._show_board(read, prefix="單次辨識")
        self.status.set(
            f"辨識完成｜最低信心 {read.confidence:.2f}｜狀態 {read.terminal or '進行中'}"
        )

    def start(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        settings = self._settings()
        if settings is None:
            return
        self.stop_event.clear()
        self.auto_hidden = self._window_overlaps_grid(settings.geometry)
        self.status.set("正在載入 V44…")
        if self.auto_hidden:
            self.root.iconify()
            self.root.update_idletasks()
        self.worker = threading.Thread(target=self._run, args=(settings,), daemon=True)
        self.worker.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.status.set("正在停止…")

    def _run(self, settings: RunSettings) -> None:
        try:
            agent, _ = load_replay_deployment(
                MODEL_CONFIG, device="cpu", cpu_threads=CPU_THREADS
            )
            engine = LiveDecisionEngine(agent)
            recognizer = HdSkinRecognizer(
                colour_tolerance=settings.tolerance,
                theme=settings.theme,
            )
            self.events.put(("status", f"V44 純無旗最大展開策略｜CPU {CPU_THREADS} 執行緒"))
            started: float | None = None
            effective_no_guess = settings.mode == "no_guess"
            previous = None
            previous_action = None
            unchanged = 0
            low_confidence = 0
            solver_errors = 0
            resets = 0
            actions = 0
            reveal_actions = 0
            flag_actions = 0
            chord_actions = 0
            lost_confirmations = 0
            won_confirmations = 0

            while not self.stop_event.is_set():
                cycle_started = time.perf_counter()
                read = recognizer.read_screen(settings.geometry, total_mines=settings.mines)
                now = time.perf_counter()
                if read.errors:
                    low_confidence += 1
                    self.events.put(("status", f"畫面信心不足，重讀 {low_confidence}/{MAX_VISUAL_RETRIES}"))
                    if low_confidence >= MAX_VISUAL_RETRIES:
                        self.events.put(("fatal", "畫面辨識持續低信心，已安全停止，沒有亂點。"))
                        break
                    self.stop_event.wait(0.05)
                    continue
                low_confidence = 0
                if read.start_cell is not None and settings.mode != "multiplayer":
                    if not effective_no_guess:
                        effective_no_guess = True
                        self.events.put(("mode", "no_guess"))
                        self.events.put(("status", "看到綠色 X，已自動切換為無猜策略"))

                reset = detect_region_reset(previous, read.observation)
                if reset and settings.mode == "multiplayer":
                    resets += 1
                    engine.reset()
                    unchanged = 0
                    previous_action = None
                    self.events.put(("status", f"偵測到區域重置，繼續解第 {resets + 1} 區"))

                if read.terminal == "won":
                    won_confirmations += 1
                    if won_confirmations < MAX_TERMINAL_CONFIRMATIONS:
                        self.events.put(("status", "疑似勝利，等待網站畫面再次確認"))
                        self.stop_event.wait(0.12)
                        continue
                    elapsed = max(0.001, now - (started or now))
                    three_bv = _three_bv_from_completed(read, settings.mines)
                    summary = _metric_text(actions, elapsed, three_bv, resets)
                    self.events.put(("metrics", summary))
                    self.events.put(("board", (read, "完成")))
                    self.events.put(("status", "勝利已確認；等待網站完成對局結算…"))
                    self.stop_event.wait(WIN_SETTLE_SECONDS)
                    self.events.put(("status", "已辨識勝利並保留網站結算時間"))
                    break
                won_confirmations = 0
                if read.terminal == "lost":
                    if settings.mode == "multiplayer":
                        self.events.put(("status", "看到踩雷畫面；等待網站重置該區域"))
                        previous = read.observation
                        self.stop_event.wait(max(0.12, self.current_delay_ms / 1000))
                        continue
                    lost_confirmations += 1
                    if lost_confirmations < MAX_TERMINAL_CONFIRMATIONS:
                        self.events.put(("board", (read, "疑似失敗畫面，尚未停止")))
                        self.events.put((
                            "status",
                            f"疑似失敗，重新確認 {lost_confirmations}/{MAX_TERMINAL_CONFIRMATIONS}",
                        ))
                        self.stop_event.wait(0.12)
                        continue
                    self.events.put(("board", (read, "失敗")))
                    self.events.put(("status", "連續兩次辨識為失敗，已停止"))
                    break
                lost_confirmations = 0

                if previous_action is not None and previous == read.observation:
                    unchanged += 1
                else:
                    unchanged = 0
                if unchanged >= MAX_UNCHANGED_RETRIES:
                    self.events.put(("fatal", "點擊後棋盤長時間沒有變化；可能校準錯誤或網頁失焦，已安全停止。"))
                    break
                if unchanged:
                    self.events.put(("status", f"等待網頁更新 {unchanged}/{MAX_UNCHANGED_RETRIES}"))
                    self.stop_event.wait(WEB_UPDATE_POLL_SECONDS)
                    continue

                revealed = sum(cell >= 0 for row in read.observation for cell in row)
                start_cell = read.start_cell if revealed == 0 else None
                if revealed == 0 and (effective_no_guess or settings.mode == "multiplayer") and start_cell is None:
                    self.events.put(("board", (read, "尚未辨識到綠色 X 起始格，持續重讀")))
                    previous = read.observation
                    previous_action = None
                    self.stop_event.wait(0.05)
                    continue

                if start_cell is not None:
                    action = LiveAction(
                        "reveal", start_cell, "網站指定的綠色 X 無猜起始格", 0.0, True
                    )
                else:
                    allow_guess = settings.mode == "multiplayer" or not effective_no_guess
                    action = engine.next_action(
                        read.observation,
                        settings.mines,
                        allow_guess=allow_guess,
                        allowed=read.allowed,
                    )
                if action.action == "stuck" or action.coord is None:
                    if "invalid" in action.reason or "contradiction" in action.reason:
                        solver_errors += 1
                        self.events.put(("status", f"盤面矛盾，快速確認 {solver_errors}/{MAX_CONTRADICTION_RETRIES}"))
                        if solver_errors >= MAX_CONTRADICTION_RETRIES:
                            self.events.put(("fatal", f"連續兩次相同矛盾，立即停止：{action.reason}"))
                            break
                        self.stop_event.wait(0.02)
                        continue
                    if effective_no_guess or settings.mode == "multiplayer":
                        self.events.put(("board", (read, f"等待：{action.reason}")))
                        previous = read.observation
                        previous_action = None
                        self.stop_event.wait(0.05)
                        continue
                    self.events.put(("fatal", f"模型無可用動作：{action.reason}"))
                    break

                row, col = action.coord
                solver_errors = 0
                screen_coord = settings.geometry.cell_center(row, col)
                probability = "—" if action.mine_probability is None else f"{action.mine_probability:.4f}"
                label = (
                    f"{action.action} ({row},{col})｜雷率 {probability}｜"
                    f"exact={action.exact}｜{action.reason}"
                )
                self.events.put(("board", (read, label)))
                self.mouse.click(screen_coord, button="right" if action.action == "flag" else "left")
                self.mouse.move((settings.geometry.left - 12, settings.geometry.top - 12))
                self.stop_event.wait(HOVER_SETTLE_SECONDS)
                if started is None:
                    started = time.perf_counter()
                actions += 1
                reveal_actions += action.action == "reveal"
                flag_actions += action.action == "flag"
                chord_actions += action.action == "chord"
                elapsed = max(0.001, time.perf_counter() - started)
                processing_ms = (time.perf_counter() - cycle_started) * 1000
                detail = (
                    f"點擊 {actions}（開 {reveal_actions}/旗 {flag_actions}/Chord {chord_actions}）"
                    f"｜時間 {elapsed:.2f}s｜本步辨識+運算 {processing_ms:.0f}ms"
                    f"｜額外延遲 {self.current_delay_ms:.0f}ms｜網站 IOE 待結算｜重置 {resets}"
                )
                self.events.put(("metrics", detail))
                previous = read.observation
                previous_action = action
                self.stop_event.wait(max(0.0, self.current_delay_ms / 1000))
        except Exception as error:
            self.events.put(("fatal", f"桌面代理錯誤：{type(error).__name__}: {error}"))
        finally:
            self.stop_event.set()
            if self.auto_hidden:
                self.events.put(("restore", None))

    def _show_board(self, read: BoardRead, prefix: str) -> None:
        allowed = sum(value for row in read.allowed for value in row)
        text = (
            f"[{time.strftime('%H:%M:%S')}] {prefix}\n"
            f"confidence={read.confidence:.2f}, clickable={allowed}, "
            f"start={read.start_cell or '-'}, terminal={read.terminal or '-'}\n{format_observation(read.observation)}\n\n"
        )
        self.log.insert("end", text)
        self.log.see("end")

    def _poll_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "status":
                    self.status.set(str(payload))
                elif kind == "metrics":
                    self.metrics.set(str(payload))
                elif kind == "mode":
                    self.mode.set(str(payload))
                elif kind == "board":
                    read, prefix = payload  # type: ignore[misc]
                    self._show_board(read, prefix)
                elif kind == "fatal":
                    self.status.set(str(payload))
                    self.stop_event.set()
                elif kind == "restore":
                    self._restore_window()
        except queue.Empty:
            pass
        self.root.after(60, self._poll_events)

    def _poll_hotkeys(self) -> None:
        user32 = ctypes.windll.user32
        for key, callback in ((VK_F8, self.start), (VK_F9, self.stop)):
            down = bool(user32.GetAsyncKeyState(key) & 0x8000)
            if down and not self._key_down[key]:
                callback()
            self._key_down[key] = down
        self.root.after(80, self._poll_hotkeys)

    def _window_overlaps_grid(self, geometry: GridGeometry) -> bool:
        self.root.update_idletasks()
        window_left = self.root.winfo_rootx()
        window_top = self.root.winfo_rooty()
        window_right = window_left + self.root.winfo_width()
        window_bottom = window_top + self.root.winfo_height()
        return not (
            window_right <= geometry.left
            or window_left >= geometry.right
            or window_bottom <= geometry.top
            or window_top >= geometry.bottom
        )

    def _restore_window(self) -> None:
        self.auto_hidden = False
        self.root.deiconify()
        self.root.lift()

    def _close(self) -> None:
        self.stop_event.set()
        self.root.destroy()



def _three_bv_from_completed(read: BoardRead, mines: int) -> int | None:
    mine_positions = {
        (row, col)
        for row, values in enumerate(read.observation)
        for col, value in enumerate(values)
        if value in (UNKNOWN, FLAGGED)
    }
    if len(mine_positions) != mines:
        return None
    game = MinesweeperGame(
        len(read.observation[0]),
        len(read.observation),
        mines,
        mine_positions=mine_positions,
    )
    return calculate_3bv(game).value


def _metric_text(actions: int, elapsed: float, three_bv: int | None, resets: int) -> str:
    if three_bv is None or actions == 0:
        return f"點擊 {actions}｜時間 {elapsed:.2f}s｜3BV/s —｜網站 IOE —｜重置 {resets}"
    speed = three_bv_per_second(three_bv, elapsed)
    ioe = index_of_efficiency(three_bv, actions)
    return (
        f"點擊 {actions}｜時間 {elapsed:.2f}s｜3BV {three_bv}｜"
        f"3BV/s {speed:.2f}｜網站 IOE {ioe:.3f}（{ioe * 100:.1f}%）｜重置 {resets}"
    )


def main() -> None:
    root = tk.Tk()
    DesktopBotApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
