"""Offline native Minesweeper, sharing V44's public-observation decision engine."""
from __future__ import annotations

from .windows_coordinates import enable_dpi_awareness
enable_dpi_awareness()

from dataclasses import asdict
import json
import os
from pathlib import Path
import queue
import secrets
import threading
import tkinter as tk
from tkinter import filedialog, font as tkfont, messagebox, ttk

from PIL import ImageEnhance, ImageTk

from .game import FLAGGED, UNKNOWN, GameStatus, MinesweeperGame
from .local_game import LocalGameSession, TERMINAL
from .local_skin import tile_image
from .local_feedback import build_feedback, save_feedback
from .local_replay import ReplayTimeline

from .runtime_paths import resource_root, feedback_directory

PROJECT_ROOT = resource_root()
MODEL_CONFIG = PROJECT_ROOT / "models" / "replay-agent-v44.json"
FEEDBACK_DIRECTORY = feedback_directory()
BG, PANEL, FG, MUTED, ACCENT = "#111820", "#1b2530", "#e4edf5", "#9aabba", "#62dcc1"
UI_SCALE = 1.5
BASE_CELL_SIZE = 28


def px(value):
    """Convert logical UI pixels to the fixed 150% display size."""
    if isinstance(value, tuple):
        return tuple(px(part) for part in value)
    return round(value * UI_SCALE)


PRESETS = {"初級": (9, 9, 10), "中級": (16, 16, 40),
           "高級": (16, 30, 99), "地獄": (20, 30, 130)}


def preview_cells(observation, coord):
    """Shade only unrevealed, unflagged neighbours of a revealed cell."""
    if coord is None:
        return set()
    row, col = coord
    if observation[row][col] < 0:
        return set()
    return {(r, c) for r in range(max(0, row - 1), min(len(observation), row + 2))
            for c in range(max(0, col - 1), min(len(observation[0]), col + 2))
            if observation[r][c] == UNKNOWN}


def _run_ai_worker(requests, results):
    """Background computation owns queues only, never a Tk app or widget."""
    agent = None
    engine = None
    engine_token = None
    model_info = None
    while True:
        request = requests.get()
        if request is None:
            return
        if isinstance(request, dict):
            try:
                from .replay_analysis import assess_move
                def predict_risks(observation, mines):
                    nonlocal agent, model_info
                    if agent is None:
                        from .replay_deployment import load_replay_deployment
                        agent, config = load_replay_deployment(MODEL_CONFIG, device="cpu",
                            cpu_threads=max(2, min(8, os.cpu_count() or 2)))
                        model_info = dict(name="V44", config=asdict(config))
                    return agent.risk_predictor.predict(observation, mines)
                assessment = assess_move(request["observation"], request["mines"], request["action"],
                                         predict_risks=predict_risks)
                results.put(dict(key=request["key"], text=assessment.text))
            except Exception as error:
                results.put(dict(key=request["key"], text=f"分析失敗：{type(error).__name__}：{error}"))
            continue
        token, revision, observation, mines, allow_guess, use_flags = request
        try:
            from .desktop_live import LiveDecisionEngine
            if agent is None:
                from .replay_deployment import load_replay_deployment
                agent, config = load_replay_deployment(MODEL_CONFIG, device="cpu",
                                                 cpu_threads=max(2, min(8, os.cpu_count() or 2)))
                model_info = dict(name="V44", config=asdict(config) if config is not None else None)
            if engine_token != token:
                engine = LiveDecisionEngine(agent, use_flags=use_flags)
                engine_token = token
            action = engine.next_action(observation, mines, allow_guess=allow_guess)
            results.put((token, revision, action, None, model_info))
        except Exception as error:
            results.put((token, revision, None, f"{type(error).__name__}: {error}"))


class LocalMinesweeperApp:
    def __init__(self, root: tk.Tk, *, feedback_directory: Path = FEEDBACK_DIRECTORY):
        self.root = root
        self.feedback_directory = Path(feedback_directory)
        self.feedback_path = None
        self.model_info = None
        self.replay = None
        self.replay_index = 0
        self.replay_after = None
        self.press_consumed = False
        self.replay_info = tk.StringVar()
        self.replay_reason = tk.StringVar()
        self.replay_delay = tk.StringVar(value="0.5")
        self.analysis_text = tk.StringVar(value="選擇一步，再按「分析這一步」。")
        self.analysis_cache = {}
        self.analysis_epoch = 0
        self.analysis_pending = None
        self.input_status = tk.StringVar(value="未開格按下即開 · 按住數字查看九宮格")
        root.title("本機踩地雷 · V44 · 150%")
        root.configure(bg=BG)
        width, height = min(px(1270), root.winfo_screenwidth() - 80), min(px(1000), root.winfo_screenheight() - 100)
        root.geometry(f"{width}x{height}+40+40")
        root.minsize(min(px(1240), width), min(px(940), height))
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont"):
            tkfont.nametofont(name).configure(family="Microsoft JhengHei UI", size=-px(14))
        self.rows, self.cols, self.mines = tk.IntVar(value=20), tk.IntVar(value=30), tk.IntVar(value=130)
        self.first_zero = tk.BooleanVar(value=True)
        self.show_action_marker = tk.BooleanVar(value=False)
        self.seed_input = tk.StringVar()
        self.cell_pixels = px(BASE_CELL_SIZE)
        self.delay = tk.IntVar(value=100)
        self.allow_guess, self.use_flags = tk.BooleanVar(value=True), tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="準備開始")
        self.ai_status = tk.StringVar(value="尚未啟動 · 純無旗策略")
        self.board_info = tk.StringVar()
        self.stat_vars = {key: tk.StringVar(value="—") for key in ("time", "bv", "speed", "clicks", "efficiency", "progress")}
        self.seed = 0
        self.session: LocalGameSession
        self.running = False
        self.busy = False
        self.closed = False
        self.token = 0
        self.ai_after = None
        self.fit_after = None
        self.requests: queue.Queue = queue.Queue()
        self.results: queue.Queue = queue.Queue()
        self.pressed: set[int] = set()
        self.press_coord = None
        self.combo = False
        self.press_consumed = False
        self.shaded = set()
        self.tile_refs = {}
        self.items = {}
        self.displayed = {}
        self._build_ui()
        self.new_game()
        self.worker = threading.Thread(target=_run_ai_worker,
                                       args=(self.requests, self.results),
                                       daemon=True, name="local-v44")
        self.worker.start()
        root.bind("<F2>", lambda _: self.new_game())
        root.bind("<F8>", lambda _: self.start_ai())
        root.bind("<F9>", lambda _: self.stop_ai())
        root.bind("<Escape>", lambda _: self.stop_ai())
        root.bind("<FocusOut>", self._focus_out)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.poll_after = root.after(25, self._poll)

    def _label(self, parent, text=None, *, var=None, size=14, colour=FG, bold=False, **kwargs):
        return tk.Label(parent, text=text, textvariable=var, bg=parent.cget("bg"), fg=colour,
                        font=("Microsoft JhengHei UI", -px(size), "bold" if bold else "normal"), **kwargs)

    def _button(self, parent, text, command, *, primary=False):
        return tk.Button(parent, text=text, command=command, relief="flat", bd=0,
                         bg=ACCENT if primary else "#2b3a49", fg=BG if primary else FG,
                         activebackground="#8ee8d4" if primary else "#3d5164",
                         activeforeground=BG if primary else FG, padx=px(13), pady=px(8), cursor="hand2")

    def _check(self, parent, text, variable, command=None):
        return ttk.Checkbutton(parent, text=text, variable=variable, command=command,
                               style="Game.TCheckbutton")

    def _build_ui(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TScrollbar", background="#354454", troughcolor=PANEL,
                        arrowcolor=MUTED, bordercolor=PANEL, lightcolor=PANEL, darkcolor=PANEL,
                        arrowsize=px(14), borderwidth=px(1))
        style.map("TScrollbar", background=[("disabled", "#354454"), ("active", "#4b6073"),
                                                   ("!active", "#354454")],
                  arrowcolor=[("disabled", "#718495"), ("!disabled", MUTED)])
        style.configure("TCombobox", fieldbackground=BG, background="#354454",
                        foreground=FG, arrowcolor=MUTED, bordercolor=PANEL)
        style.map("TCombobox", fieldbackground=[("readonly", BG)], foreground=[("readonly", FG)])
        style.configure("Game.TCheckbutton", background=PANEL, foreground=FG,
                        font="TkDefaultFont", indicatorsize=px(12),
                        indicatormargin=(0, 0, px(6), 0),
                        indicatorbackground=BG, indicatorforeground=FG)
        style.map("Game.TCheckbutton", background=[("active", PANEL)],
                  indicatorbackground=[("selected", ACCENT), ("!selected", BG)],
                  indicatorforeground=[("selected", BG), ("!selected", FG)])
        style.configure("Game.TSpinbox", fieldbackground=BG, foreground=FG,
                        background="#354454", arrowcolor=MUTED, arrowsize=px(10),
                        padding=px(1), borderwidth=px(1), insertcolor=FG)
        outer = tk.Frame(self.root, bg=BG, padx=px(22), pady=px(18))
        outer.pack(fill="both", expand=True)
        header = tk.Frame(outer, bg=BG)
        header.pack(fill="x")
        self._label(header, "本機踩地雷", size=28, bold=True).pack(side="left")
        self._label(header, "  OFFLINE / V44", size=13, colour=ACCENT).pack(side="left", pady=px((8, 0)))
        self._button(header, "匯出紀錄", self.export).pack(side="right", padx=px(8))
        self._button(header, "載入重播", self.load_replay).pack(side="right", padx=px(4))
        self._button(header, "重播本局", self.replay_current).pack(side="right", padx=px(4))
        self._label(outer, "自己的棋盤，隨時開局。手動遊玩或交給現有的 V44。", colour=MUTED).pack(anchor="w", pady=px((6, 14)))

        settings = tk.Frame(outer, bg=PANEL, padx=px(12), pady=px(10))
        settings.pack(fill="x")
        row1 = tk.Frame(settings, bg=PANEL)
        row1.pack(fill="x")
        for label, spec in PRESETS.items():
            self._button(row1, label, lambda s=spec: self._preset(s)).pack(side="left", padx=px((0, 5)))
        for label, var, maximum in (("列", self.rows, 60), ("欄", self.cols, 80), ("雷", self.mines, 4799)):
            self._label(row1, label, colour=MUTED).pack(side="left", padx=px((10, 4)))
            ttk.Spinbox(row1, from_=2 if label != "雷" else 1, to=maximum, textvariable=var,
                        width=5, font="TkDefaultFont", style="Game.TSpinbox").pack(side="left")
        self._button(row1, "套用自訂", self.new_game).pack(side="left", padx=px((10, 0)))
        self._button(row1, "新的一局  F2", self.new_game, primary=True).pack(side="left", padx=px((10, 0)))
        row2 = tk.Frame(settings, bg=PANEL)
        row2.pack(fill="x", pady=px((9, 0)))
        self._check(row2, "首點展開空白區", self.first_zero).pack(side="left")
        self._label(row2, "種子", colour=MUTED).pack(side="left", padx=px((16, 5)))
        tk.Entry(row2, textvariable=self.seed_input, width=15, bg=BG, fg=FG,
                 insertbackground=FG, relief="flat").pack(side="left")
        self._label(row2, "留白即隨機", colour=MUTED, size=12).pack(side="left", padx=px(7))
        self._label(row2, "顯示比例 150%", colour=ACCENT).pack(side="left", padx=px((16, 5)))
        self.marker_check = self._check(row2, "顯示點擊綠框", self.show_action_marker, self._render)
        self.marker_check.pack(side="left", padx=px((16, 0)))

        content = tk.Frame(outer, bg=BG)
        content.pack(fill="both", expand=True, pady=px((16, 0)))
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=1)
        board_panel = tk.Frame(content, bg=PANEL, padx=px(10), pady=px(10))
        board_panel.grid(row=0, column=0, sticky="nsew", padx=px((0, 16)))
        self._label(board_panel, var=self.board_info, colour=ACCENT, bold=True).pack(anchor="w", pady=px((0, 8)))
        self.replay_panel = tk.Frame(board_panel, bg=PANEL)
        replay_buttons = tk.Frame(self.replay_panel, bg=PANEL)
        replay_buttons.pack(fill="x")
        self._button(replay_buttons, "開局", lambda: self.seek_replay(0)).pack(side="left")
        self._button(replay_buttons, "上一步", lambda: self.step_replay(-1)).pack(side="left", padx=px(4))
        self.replay_play_button = self._button(replay_buttons, "播放", self.toggle_replay, primary=True)
        self.replay_play_button.pack(side="left")
        self._button(replay_buttons, "下一步", lambda: self.step_replay(1)).pack(side="left", padx=px(4))
        self._label(replay_buttons, "每步", colour=MUTED).pack(side="left", padx=px((8, 3)))
        ttk.Combobox(replay_buttons, textvariable=self.replay_delay, values=("0.1", "0.25", "0.5", "1", "2"),
                     width=4, state="readonly", font="TkDefaultFont").pack(side="left")
        self._label(replay_buttons, "秒", colour=MUTED).pack(side="left", padx=px(3))
        self._button(replay_buttons, "返回對局", self.leave_replay).pack(side="right")
        self.replay_scale = tk.Scale(self.replay_panel, from_=0, to=1, resolution=1, showvalue=False,
            command=self._slider_replay, orient="horizontal", bg=PANEL, fg=FG, troughcolor=BG,
            activebackground=ACCENT, highlightthickness=0, bd=0, width=px(15), sliderlength=px(22))
        self.replay_scale.pack(fill="x", pady=px((5, 2)))
        self._label(self.replay_panel, var=self.replay_info, size=12, colour=ACCENT).pack(anchor="w")
        self._label(self.replay_panel, var=self.replay_reason, size=12, colour=MUTED,
                    justify="left", wraplength=px(850)).pack(anchor="w", pady=px((2, 8)))
        board_area = tk.Frame(board_panel, bg=PANEL)
        self.board_area = board_area
        board_area.pack(fill="both", expand=True)
        board_area.rowconfigure(0, weight=1)
        board_area.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(board_area, bg=PANEL, highlightthickness=0, bd=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.yscroll = ttk.Scrollbar(board_area, orient="vertical", command=self.canvas.yview)
        self.yscroll.grid(row=0, column=1, sticky="ns")
        self.xscroll = ttk.Scrollbar(board_area, orient="horizontal", command=self.canvas.xview)
        self.xscroll.grid(row=1, column=0, sticky="ew")
        self.canvas.configure(xscrollcommand=self.xscroll.set, yscrollcommand=self.yscroll.set)
        for button in (1, 2, 3):
            self.canvas.bind(f"<ButtonPress-{button}>", self._press)
            self.canvas.bind(f"<ButtonRelease-{button}>", self._release)
        self.canvas.bind("<B1-Motion>", self._drag_preview)
        self.canvas.bind("<Leave>", lambda _: self._set_preview(None))
        self.canvas.bind("<MouseWheel>", self._wheel)
        self._label(board_panel, var=self.input_status, colour=MUTED, size=12).pack(anchor="w", pady=px((8, 0)))

        sidebar_shell = tk.Frame(content, bg=BG, width=px(294))
        sidebar_shell.grid(row=0, column=1, sticky="ns")
        sidebar_shell.grid_propagate(False)
        sidebar_shell.rowconfigure(0, weight=1)
        sidebar_shell.columnconfigure(0, weight=1)
        side_canvas = tk.Canvas(sidebar_shell, bg=BG, highlightthickness=0, width=px(278))
        side_canvas.grid(row=0, column=0, sticky="nsew")
        side_scroll = ttk.Scrollbar(sidebar_shell, orient="vertical", command=side_canvas.yview)
        side_scroll.grid(row=0, column=1, sticky="ns")
        side_canvas.configure(yscrollcommand=side_scroll.set)
        sidebar = tk.Frame(side_canvas, bg=BG)
        side_window = side_canvas.create_window(0, 0, anchor="nw", window=sidebar)
        sidebar.columnconfigure(0, weight=1)
        sidebar.bind("<Configure>", lambda _: side_canvas.configure(scrollregion=side_canvas.bbox("all")))
        side_canvas.bind("<Configure>", lambda e: side_canvas.itemconfigure(side_window, width=e.width))
        self.side_canvas = side_canvas
        stats = tk.Frame(sidebar, bg=PANEL, padx=px(14), pady=px(10))
        stats.grid(row=0, column=0, sticky="ew")
        self._label(stats, "本局耗時", size=13, colour=MUTED).pack(anchor="w")
        timer = tk.Frame(stats, bg=PANEL)
        timer.pack(fill="x", pady=px((4, 6)))
        self._label(timer, var=self.stat_vars["time"], size=32, bold=True).pack(side="left")
        self._label(timer, "秒", colour=MUTED).pack(side="left", padx=px(8), pady=px((12, 0)))
        for label, key in (("3BV", "bv"), ("3BV/s", "speed"), ("點擊", "clicks"), ("效率", "efficiency")):
            line = tk.Frame(stats, bg=PANEL)
            line.pack(fill="x", pady=px(1))
            self._label(line, label, colour=MUTED, size=14).pack(side="left")
            self._label(line, var=self.stat_vars[key], size=16, bold=True).pack(side="right")
        self._label(stats, var=self.stat_vars["progress"], colour=ACCENT, size=12).pack(anchor="w", pady=px((10, 0)))
        self._label(stats, "點擊＝有效＋多餘", colour=MUTED,
                    size=12, justify="left").pack(anchor="w", pady=px((4, 0)))

        ai = tk.Frame(sidebar, bg=PANEL, padx=px(14), pady=px(12))
        ai.grid(row=1, column=0, sticky="ew", pady=px((12, 0)))
        self.ai_panel = ai
        self.analysis_panel = tk.Frame(sidebar, bg=PANEL, padx=px(14), pady=px(12))
        self._label(self.analysis_panel, "重播動作分析", size=16, bold=True).pack(anchor="w")
        self.analysis_button = self._button(self.analysis_panel, "分析這一步", self.analyse_replay_step, primary=True)
        self.analysis_button.pack(fill="x", pady=px((10,8)))
        self._label(self.analysis_panel, var=self.analysis_text, size=13, wraplength=px(246),
                    justify="left").pack(anchor="w")
        self._label(self.analysis_panel, "本機分析 · 結果快取 · 不訓練", size=12, colour=MUTED,
                    wraplength=px(246)).pack(anchor="w", pady=px((10,0)))
        self._label(ai, "V44 自動遊玩", size=16, bold=True).pack(anchor="w")
        buttons = tk.Frame(ai, bg=PANEL)
        buttons.pack(fill="x", pady=px((10, 8)))
        self._button(buttons, "開始 F8", self.start_ai, primary=True).pack(side="left")
        self._button(buttons, "停止 F9", self.stop_ai).pack(side="right")
        self._button(ai, "執行一步", lambda: self.start_ai(single=True)).pack(fill="x")
        self._check(ai, "允許必要的猜測", self.allow_guess, self.stop_ai).pack(anchor="w", pady=px((7, 0)))
        self._check(ai, "使用插旗＋連開策略", self.use_flags, self.stop_ai).pack(anchor="w")
        self._label(ai, "每步額外延遲（毫秒）", colour=MUTED, size=12).pack(anchor="w", pady=px((6, 0)))
        tk.Scale(ai, from_=0, to=1000, resolution=10, variable=self.delay, orient="horizontal",
                 bg=PANEL, fg=FG, troughcolor=BG, activebackground=ACCENT,
                 highlightthickness=0, bd=0, length=px(250), width=px(15), sliderlength=px(30)).pack(fill="x")
        self._label(ai, var=self.ai_status, colour=MUTED, size=12, wraplength=px(246), justify="left").pack(anchor="w", pady=px((4, 0)))
        self._label(outer, var=self.status, colour=MUTED, size=13, anchor="w").pack(fill="x", pady=px((12, 0)))

    def _preset(self, spec):
        self.rows.set(spec[0])
        self.cols.set(spec[1])
        self.mines.set(spec[2])
        self.new_game()

    def new_game(self):
        if self.replay is not None:
            self.leave_replay()
        try:
            rows, cols, mines = self.rows.get(), self.cols.get(), self.mines.get()
            seed = int(self.seed_input.get()) if self.seed_input.get().strip() else secrets.randbits(32)
            if not (2 <= rows <= 60 and 2 <= cols <= 80 and 1 <= mines < rows * cols):
                raise ValueError("列需為 2–60、欄為 2–80；雷數須介於 1 與總格數減 1。")
            if self.first_zero.get() and mines > rows * cols - min(rows, 3) * min(cols, 3):
                raise ValueError("雷數過多，無法保證首點空白；請減少雷數或取消首點展開。")
        except (ValueError, tk.TclError) as error:
            messagebox.showerror("請檢查盤面設定", str(error), parent=self.root)
            return
        self.stop_ai(quiet=True)
        self.seed = seed
        self.input_status.set("未開格按下即開 · 按住數字查看九宮格")
        self.feedback_path = None
        self.model_info = None
        self.canvas.delete("all")
        self.items.clear()
        self.displayed.clear()
        self.session = LocalGameSession(MinesweeperGame(cols, rows, mines, seed=seed,
                                                     first_click_zero=self.first_zero.get()))
        self._clear_press()
        self._resize_board()
        self.ai_status.set("尚未啟動 · " + ("插旗＋連開策略" if self.use_flags.get() else "純無旗策略"))
        self.status.set(f"準備開始 · 種子 {seed} · 首點安全" + ("並展開空白區" if self.first_zero.get() else ""))
        self._update_stats()

    def _resize_board(self):
        self.stop_ai(quiet=True)
        self._clear_press()
        self.canvas.delete("all")
        self.items.clear()
        self.displayed.clear()
        size = self.cell_pixels
        # Warm image/font resources before the first click; keep this UI-thread cache.
        for value in (UNKNOWN, FLAGGED, *range(9), "mine", "exploded"):
            self._tile(value, False)
        self._tile(UNKNOWN, True)
        game = self.view_game
        for r in range(game.height):
            for c in range(game.width):
                self.items[r, c] = self.canvas.create_image(c * size, r * size, anchor="nw")
        self.canvas.configure(scrollregion=(0, 0, game.width * size, game.height * size))
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)
        self._fit_window_for_board()
        self._render()

    def _fit_window_for_board(self):
        # Keep tile size exactly as selected; resize only the window.
        # Even a smaller preset reserves room for the full 20 x 30 evil board.
        if self.fit_after is not None:
            self.root.after_cancel(self.fit_after)
            self.fit_after = None
        self.root.update_idletasks()
        if not self.canvas.winfo_ismapped():
            # Unmapped Tk widgets report 1 x 1, not their actual available size.
            if not self.closed:
                self.fit_after = self.root.after(50, self._fit_window_for_board)
            return
        size = self.cell_pixels
        overhead_w = self.root.winfo_width() - self.canvas.winfo_width()
        overhead_h = self.root.winfo_height() - self.canvas.winfo_height()
        screen_w = self.root.winfo_screenwidth() - 80
        screen_h = self.root.winfo_screenheight() - 100
        minimum_w = min(screen_w, overhead_w + 30 * size + px(4))
        minimum_h = min(screen_h, overhead_h + 20 * size + px(4))
        wanted_w = min(screen_w, max(self.root.winfo_width(), minimum_w,
                                    overhead_w + self.view_game.width * size + px(4)))
        wanted_h = min(screen_h, max(self.root.winfo_height(), minimum_h,
                                    overhead_h + self.view_game.height * size + px(4)))
        self.root.minsize(minimum_w, minimum_h)
        self.root.geometry(f"{wanted_w}x{wanted_h}")
        self.root.update_idletasks()

    @property
    def view_game(self):
        return self.replay.board_at(self.replay_index) if self.replay is not None else self.session.game

    def _tile(self, value, shaded):
        key = (value, shaded)
        if key not in self.tile_refs:
            image = tile_image(value, self.cell_pixels)
            if shaded:
                image = ImageEnhance.Brightness(image).enhance(0.52)
            self.tile_refs[key] = ImageTk.PhotoImage(image, master=self.root)
        return self.tile_refs[key]

    def _render(self):
        game = self.view_game
        obs = game.observation
        for coord, item in self.items.items():
            r, c = coord
            value = obs[r][c]
            if game.status in TERMINAL and value != FLAGGED and game.is_mine(coord):
                # Completion flags are visual; keep recorded player actions intact.
                value = (FLAGGED if game.status == GameStatus.WON else
                         "exploded" if coord == game.exploded else "mine")
            key = (value, coord in self.shaded)
            if self.displayed.get(coord) == key:
                continue
            self.canvas.itemconfigure(item, image=self._tile(*key))
            self.displayed[coord] = key
        self.canvas.delete("action-marker")
        if self.show_action_marker.get():
            action = (self.replay.steps[self.replay_index].action if self.replay is not None else
                      self.session.history[-1] if self.session.history else None)
            if action is not None:
                r, c = action["row"], action["col"]
                size = self.cell_pixels
                # Keep the border outside the recognizer's sampled tile interior.
                self.canvas.create_rectangle(c*size+1, r*size+1, (c+1)*size-1, (r+1)*size-1,
                                             outline=ACCENT, width=1, tags="action-marker")

    def _coord(self, event):
        size = self.cell_pixels
        r, c = int(self.canvas.canvasy(event.y) // size), int(self.canvas.canvasx(event.x) // size)
        game = self.view_game
        return (r, c) if 0 <= r < game.height and 0 <= c < game.width else None

    def _press(self, event):
        if self.running or (self.replay is None and self.session.game.status in TERMINAL):
            return
        coord = self._coord(event)
        if coord is None:
            return
        self.canvas.focus_set()
        self.pressed.add(event.num)
        if len(self.pressed) == 1:
            self.press_coord = coord
            self.press_consumed = False
        if (event.num == 1 and self.pressed == {1} and self.replay is None and not self.combo
                and self.session.game.observation[coord[0]][coord[1]] == UNKNOWN):
            # Unknown cells open on press, so rapid movement before release cannot cancel them.
            self.press_consumed = True
            self._act("reveal", coord)
            return
        if {1, 3} <= self.pressed or event.num == 2:
            self.combo = True
        if 1 in self.pressed or self.combo:
            self._set_preview(coord)

    def _drag_preview(self, event):
        if 1 in self.pressed:
            self._set_preview(self._coord(event))

    def _set_preview(self, coord):
        cells = preview_cells(self.view_game.observation, coord)
        if cells != self.shaded:
            self.shaded = cells
            self._render()

    def _release(self, event):
        if event.num not in self.pressed:
            return
        coord = self._coord(event)
        self.pressed.discard(event.num)
        self._set_preview(None)
        if self.replay is not None or self.press_consumed:
            if not self.pressed:
                self.press_coord = None
                self.press_consumed = False
                self.combo = False
            return
        # A simultaneous pair is ONE chord, never a subsequent right/left click.
        if self.combo:
            if self.press_coord is not None and coord == self.press_coord:
                self._act("chord", coord)
            self.press_coord = None
            if not self.pressed:
                self.combo = False
            return
        if coord is not None and coord == self.press_coord:
            value = self.session.game.observation[coord[0]][coord[1]]
            action = "flag" if event.num == 3 else ("chord" if value >= 0 else "reveal")
            self._act(action, coord)
        elif self.press_coord is not None:
            self.input_status.set("連開已取消：按下與放開在不同格；未開格則在按下時立即開啟。")
        self.press_coord = None

    def _clear_press(self):
        self.pressed.clear()
        self.press_coord = None
        self.combo = False
        self.press_consumed = False
        self.shaded = set()
        if hasattr(self, "session"):
            self._render()

    def _focus_out(self, event):
        if event.widget == self.canvas or (event.widget == self.root and self.root.focus_displayof() is None):
            self._clear_press()

    def _wheel(self, event):
        self._clear_press()
        if event.state & 1:
            self.canvas.xview_scroll(-int(event.delta / 120), "units")
        else:
            self.canvas.yview_scroll(-int(event.delta / 120), "units")

    def _act(self, action, coord, *, source="human", decision=None):
        if self.replay is not None:
            return
        if source == "human" and self.running:
            return
        if self.session.game.status in TERMINAL:
            return
        self.session.apply(action, coord, source=source, decision=decision)
        action_name = {"reveal":"開格", "flag":"插旗／取消旗", "chord":"連開"}[action]
        changed = self.session.history[-1]["effective"]
        self.input_status.set(f"已接受 {action_name} ({coord[0]+1}, {coord[1]+1}) · "
                              + ("盤面已更新" if changed else "無盤面變化"))
        self._render()
        self._update_stats()
        game = self.session.game
        if game.status in TERMINAL:
            self.stop_ai(quiet=True)
            self.status.set(("完成！所有安全格已揭露" if game.status == GameStatus.WON else "踩到地雷了")
                            + f" · {self.session.elapsed:.3f} 秒 · 按 F2 再來一局 · 種子 {self.seed}")
            result_text = "本局已完成" if game.status == GameStatus.WON else "本局已結束"
            if game.status == GameStatus.LOST and decision is not None:
                probability = decision.get("mine_probability")
                if probability is not None:
                    result_text = ("猜測踩雷" if probability > 0 else "安全判斷與結果不符，待查")
                    result_text += f" · 落子前雷率 {probability:.2%}"
            try:
                self.feedback_path = save_feedback(
                    build_feedback(self.session, seed=self.seed, model=self.model_info),
                    self.feedback_directory)
                result_text += "\n本局紀錄已保存，供後續訓練"
            except OSError as error:
                result_text += "\n紀錄儲存失敗，可用匯出重試"
                self.status.set(f"本局結束；紀錄儲存失敗：{error}")
            self.ai_status.set(result_text)
        else:
            self.status.set(f"進行中 · 種子 {self.seed} · F9 停止 AI · 按住數字查看相鄰格")

    def _update_stats(self):
        s = self.replay.steps[self.replay_index].summary if self.replay is not None else self.session.summary()
        self.stat_vars["time"].set(f"{s['elapsed_seconds']:.3f}")
        self.stat_vars["bv"].set(str(s["three_bv"]) if s["three_bv"] is not None else "—")
        self.stat_vars["speed"].set(f"{s['three_bv_per_second']:.4f}" if s["three_bv_per_second"] is not None else "—")
        self.stat_vars["clicks"].set(f"{sum(s['effective_clicks'].values())}+{sum(s['redundant_clicks'].values())}")
        self.stat_vars["efficiency"].set(f"{s['efficiency'] * 100:.0f}%" if s["efficiency"] is not None else "—")
        self.stat_vars["progress"].set(f"已完成 3BV  {s['completed_bv']} / {s['three_bv'] if s['three_bv'] is not None else '—'}")
        names = {"ready": "待開局", "active": "進行中", "won": "已完成", "lost": "已踩雷"}
        game = self.view_game
        remaining = 0 if game.status in TERMINAL else game.remaining_mines
        self.board_info.set(f"{s['cols']} × {s['rows']}    /    剩餘雷 {remaining}    /    {names[s['status']]}")

    def start_ai(self, single=False):
        if self.replay is not None:
            self.ai_status.set("重播中；返回對局後才能啟動 AI。")
            return
        if self.session.game.status in TERMINAL:
            self.ai_status.set("本局已結束，按 F2 開新局")
            return
        if self.running or self.busy:
            return
        self._clear_press()
        self.token += 1
        self.running = not single
        self.ai_status.set("載入／計算 V44…")
        self._request_step()

    def stop_ai(self, quiet=False):
        self.running = False
        self.token += 1
        self._clear_press()
        if self.ai_after is not None:
            self.root.after_cancel(self.ai_after)
            self.ai_after = None
        if not quiet:
            self.ai_status.set("已停止 · 可手動接續，計時繼續")

    def _request_step(self):
        self.ai_after = None
        if self.closed or self.busy or self.session.game.status in TERMINAL:
            return
        self.busy = True
        # Deliberately pass no game, seed, statistics or mine positions.
        self.requests.put((self.token, self.session.revision, self.session.game.observation,
                           self.session.game.mine_count, self.allow_guess.get(), self.use_flags.get()))

    def _accept_result(self, token, revision, action, error, model_info=None):
        self.busy = False
        if self.replay is not None or token != self.token or revision != self.session.revision:
            return
        if error:
            self.stop_ai(quiet=True)
            self.ai_status.set("V44 載入／運算失敗")
            messagebox.showerror("V44 無法執行", error, parent=self.root)
            return
        if action.coord is None or action.action == "stuck":
            self.stop_ai(quiet=True)
            self.ai_status.set("沒有可執行的安全動作；可手動接續或允許猜測。\n" + action.reason)
            return
        self.model_info = model_info
        p = action.mine_probability
        risk = "未知" if p is None else f"{p:.2%}"
        self.ai_status.set(f"{'自動' if self.running else '單步'} · ({action.coord[0]+1}, {action.coord[1]+1})\n"
                           f"雷率 {risk} · {'精確推理' if action.exact else '機率估計'}")
        self._act(action.action, action.coord, source="V44",
                  decision=dict(reason=action.reason, mine_probability=p, exact=action.exact,
                                allow_guess=self.allow_guess.get(), use_flags=self.use_flags.get()))
        if self.running:
            self.ai_after = self.root.after(max(1, self.delay.get()), self._request_step)

    def _poll(self):
        if self.closed:
            return
        try:
            while True:
                result = self.results.get_nowait()
                if isinstance(result, dict):
                    self._accept_analysis(result)
                else:
                    self._accept_result(*result)
        except queue.Empty:
            pass
        self._update_stats()
        self.poll_after = self.root.after(25, self._poll)

    def replay_current(self):
        if not self.session.history:
            self.status.set("本局還沒有動作，可先開格或載入已保存的紀錄。")
            return
        self.stop_ai(quiet=True)
        try:
            timeline = ReplayTimeline.from_session(self.session, seed=self.seed)
        except (ValueError, KeyError, TypeError) as error:
            messagebox.showerror("無法重播", str(error), parent=self.root)
            return
        self.enter_replay(timeline)

    def load_replay(self):
        path = filedialog.askopenfilename(parent=self.root, title="載入重播紀錄",
            initialdir=str(self.feedback_directory), filetypes=[("JSON 紀錄", "*.json")])
        if not path:
            return
        try:
            timeline = ReplayTimeline(json.loads(Path(path).read_text(encoding="utf-8")))
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            messagebox.showerror("無法載入重播", str(error), parent=self.root)
            return
        self.enter_replay(timeline)

    def enter_replay(self, timeline):
        self.pause_replay()
        self.stop_ai(quiet=True)
        self.session.pause_clock()
        self.canvas.delete("all")
        self.items.clear()
        self.displayed.clear()
        self.replay, self.replay_index = timeline, 0
        self.analysis_epoch += 1
        self.analysis_cache = {}
        self.ai_panel.grid_remove()
        self.analysis_panel.grid(row=1, column=0, sticky="ew", pady=px((12,0)))
        self.replay_panel.pack(fill="x", before=self.board_area, pady=px((0, 6)))
        self.replay_scale.configure(to=max(1, len(timeline)))
        self.replay_scale.set(0)
        self._resize_board()
        self._show_replay_position()
        self.status.set("重播模式 · 本局計時已暫停 · 返回對局可繼續原來的棋盤")
        self.ai_status.set("重播不會重新操作、保存新對局或訓練模型。")

    def pause_replay(self):
        if self.replay_after is not None:
            self.root.after_cancel(self.replay_after)
            self.replay_after = None
        if hasattr(self, "replay_play_button"):
            self.replay_play_button.configure(text="播放")

    def _slider_replay(self, value):
        if self.replay is not None:
            index = min(len(self.replay), max(0, round(float(value))))
            if index != self.replay_index:
                self.seek_replay(index)

    def seek_replay(self, index, *, pause=True):
        if self.replay is None:
            return
        if pause:
            self.pause_replay()
        self.replay_index = min(len(self.replay), max(0, int(index)))
        self.replay_scale.set(self.replay_index)
        self._clear_press()
        self._show_replay_position()

    def _show_replay_position(self):
        self.replay_info.set(self.replay.description(self.replay_index))
        self.replay_reason.set(self.replay.reason(self.replay_index))
        self._show_analysis()
        self.input_status.set("重播中 · 拖曳進度條復原／推進")
        self._render()
        self._update_stats()

    def _show_analysis(self):
        if self.replay is None:
            return
        step = self.replay.steps[self.replay_index]
        if step.action is None:
            self.analysis_text.set("開局前沒有動作可分析。")
        else:
            from .replay_analysis import observed_effect
            cached = self.analysis_cache.get(self.replay_index)
            note = cached or ("本機分析中，完成後可分析其他步。" if self.analysis_pending else
                              "按「分析這一步」查看落子前風險。")
            self.analysis_text.set(note + "\n\n" + observed_effect(step))
        self.analysis_button.configure(state="disabled" if step.action is None or self.analysis_pending else "normal")

    def analyse_replay_step(self):
        if self.replay is None or self.replay_index == 0 or self.analysis_pending:
            return
        self.pause_replay()
        if self.replay_index in self.analysis_cache:
            self._show_analysis()
            return
        step = self.replay.steps[self.replay_index]
        previous = self.replay.board_at(self.replay_index-1)
        self.analysis_pending = (self.analysis_epoch, self.replay_index)
        # Only the preceding public board and action cross into the worker.
        self.requests.put(dict(key=self.analysis_pending, observation=previous.observation,
                               mines=previous.mine_count,
                               action={key:step.action[key] for key in ("action","row","col")}))
        self._show_analysis()

    def _accept_analysis(self, result):
        if result["key"] != self.analysis_pending:
            return
        self.analysis_pending = None
        epoch, index = result["key"]
        if self.replay is not None and epoch == self.analysis_epoch:
            self.analysis_cache[index] = result["text"]
        self._show_analysis()

    def step_replay(self, delta):
        self.seek_replay(self.replay_index + delta)

    def toggle_replay(self):
        if self.replay is None:
            return
        if self.replay_after is not None:
            self.pause_replay()
            return
        if self.replay_index == len(self.replay):
            self.seek_replay(0)
        self.replay_play_button.configure(text="暫停")
        self._play_replay_step()

    def _play_replay_step(self):
        self.replay_after = None
        if self.replay is None or self.closed:
            return
        if self.replay_index >= len(self.replay):
            self.pause_replay()
            return
        self.seek_replay(self.replay_index+1, pause=False)
        self.replay_after = self.root.after(round(float(self.replay_delay.get())*1000), self._play_replay_step)

    def leave_replay(self):
        if self.replay is None:
            return
        self.pause_replay()
        self.canvas.delete("all")
        self.items.clear()
        self.displayed.clear()
        self.replay = None
        self.analysis_epoch += 1
        self.analysis_cache = {}
        self.analysis_panel.grid_remove()
        self.ai_panel.grid()
        self.replay_panel.pack_forget()
        self.session.resume_clock()
        self._resize_board()
        self._update_stats()
        self.input_status.set("已返回原對局 · 未開格按下即開 · 按住數字查看九宮格")
        self.ai_status.set("重播已結束 · 可手動接續或按 F8 啟動 AI")
        self.status.set(f"已返回對局 · 種子 {self.seed}")

    def export(self):
        path = filedialog.asksaveasfilename(parent=self.root, title="匯出本局紀錄", defaultextension=".json",
                                          initialfile=f"minesweeper-{self.seed}.json",
                                          filetypes=[("JSON 紀錄", "*.json")])
        if not path:
            return
        payload = self.replay.payload if self.replay is not None else (build_feedback(self.session, seed=self.seed, model=self.model_info)
                   if self.session.game.status in TERMINAL else
                   dict(schema_version=1, seed=self.seed, first_click_zero=self.session.game.first_click_zero,
                        **self.session.summary(), actions=self.session.history))
        try:
            Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as error:
            messagebox.showerror("匯出失敗", str(error), parent=self.root)
            return
        self.status.set(f"已匯出本局紀錄：{path}")

    def close(self):
        if self.closed:
            return
        self.pause_replay()
        self.closed = True
        self.stop_ai(quiet=True)
        self.requests.put(None)
        self.root.after_cancel(self.poll_after)
        if self.fit_after is not None:
            self.root.after_cancel(self.fit_after)
        # Release PhotoImage Tcl resources on the UI thread while Tk is alive.
        self.canvas.delete("all")
        self.tile_refs.clear()
        self.root.destroy()


def main():
    root = tk.Tk()
    LocalMinesweeperApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
