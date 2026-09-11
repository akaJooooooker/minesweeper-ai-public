"""Opt-in native GUI integration tests: RUN_LOCAL_GUI_TESTS=1."""
import gc
import json
import threading
import tempfile
import os
from pathlib import Path
import time
import tkinter as tk
import unittest
from unittest.mock import patch

from PIL import ImageGrab
from minesweeper_ai.desktop_live import LiveAction
from minesweeper_ai.game import FLAGGED, GameStatus, MinesweeperGame
from minesweeper_ai.local_app import BASE_CELL_SIZE, UI_SCALE, LocalMinesweeperApp, preview_cells
from minesweeper_ai.local_game import LocalGameSession
from minesweeper_ai.desktop_vision import HdSkinRecognizer

@unittest.skipUnless(os.environ.get("RUN_LOCAL_GUI_TESTS") == "1", "opt-in native Tk GUI test")
class LocalGuiTests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.errors = []
        self.root.report_callback_exception = lambda *args: self.errors.append(str(args))
        self.feedback_temp = tempfile.TemporaryDirectory()
        self.app = LocalMinesweeperApp(self.root, feedback_directory=Path(self.feedback_temp.name))
        self.root.update()
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.update()

    def tearDown(self):
        self.app.close()
        self.app.worker.join(timeout=2)
        self.assertFalse(self.app.worker.is_alive())
        self.assertEqual(self.errors, [])
        # The callback closes over this TestCase. Break that cycle before a
        # later model allocation can trigger cyclic GC in a background thread.
        del self.root.report_callback_exception
        del self.app
        del self.root
        gc.collect()
        self.feedback_temp.cleanup()

    def board(self, width=4, height=4, mines=((0, 0), (0, 3))):
        self.app.canvas.delete("all")
        self.app.items.clear()
        self.app.displayed.clear()
        self.app.session = LocalGameSession(MinesweeperGame(width, height, len(mines),
                                                            mine_positions=mines))
        self.app._resize_board()
        self.root.update()

    def event(self, name, coord):
        r, c = coord
        size = self.app.cell_pixels
        self.app.canvas.event_generate(name, x=c*size+size//2, y=r*size+size//2)
        self.root.update()

    def test_hold_drag_release_preserves_flags_and_does_not_click(self):
        self.board()
        self.app._act("reveal", (1, 1))
        self.app._act("flag", (0, 0))
        before = self.app.session.game.observation
        clicks = self.app.session.clicks.total
        self.event("<ButtonPress-1>", (1, 1))
        self.assertEqual(self.app.shaded, preview_cells(before, (1, 1)))
        self.assertNotIn((0, 0), self.app.shaded)
        self.assertIn((0, 1), self.app.shaded)
        self.assertEqual(self.app.displayed[(0, 0)], (FLAGGED, False))
        self.assertEqual(self.app.session.game.observation, before)
        self.assertEqual(self.app.session.clicks.total, clicks)
        self.event("<B1-Motion>", (2, 2))
        self.assertEqual(self.app.shaded, set())
        self.event("<ButtonRelease-1>", (2, 2))
        self.assertFalse(self.app.shaded)
        self.assertEqual(self.app.session.game.observation, before)
        self.assertEqual(self.app.session.clicks.total, clicks)
        self.event("<ButtonPress-1>", (1, 1))
        self.app.canvas.event_generate("<FocusOut>")
        self.root.update()
        self.assertFalse(self.app.shaded)
        self.assertFalse(self.app.pressed)

    def test_combined_buttons_chord_exactly_once(self):
        self.board(2, 2, ((0, 0),))
        self.app._act("reveal", (1, 1))
        self.app._act("flag", (0, 0))
        before = self.app.session.clicks.total
        self.event("<ButtonPress-1>", (1, 1))
        self.event("<ButtonPress-3>", (1, 1))
        self.event("<ButtonRelease-1>", (1, 1))
        self.event("<ButtonRelease-3>", (1, 1))
        self.assertEqual(self.app.session.game.status, GameStatus.WON)
        self.assertEqual(self.app.session.clicks.total, before + 1)
        self.assertEqual(self.app.session.clicks.effective["chord"], 1)

    def test_150_percent_window_fits_evil_and_new_game_button_is_beside_custom(self):
        self.assertEqual((self.app.session.game.height, self.app.session.game.width,
                          self.app.session.game.mine_count), (20, 30, 130))
        self.assertEqual(BASE_CELL_SIZE, 28)
        self.assertEqual(UI_SCALE, 1.5)
        self.assertEqual(self.app.cell_pixels, 42)
        self.assertLessEqual(self.root.winfo_width(), 2100)
        self.assertLessEqual(self.root.winfo_height(), 1600)
        for spec in ((9, 9, 10), (20, 30, 130)):
            self.app._preset(spec)
            self.root.update()
            self.assertEqual(self.app.cell_pixels, 42)
            self.assertGreaterEqual(self.app.canvas.winfo_width(), 30 * 42)
            self.assertGreaterEqual(self.app.canvas.winfo_height(), 20 * 42)
            self.assertEqual(tuple(self.app.canvas.xview()), (0.0, 1.0))
            self.assertEqual(tuple(self.app.canvas.yview()), (0.0, 1.0))
        def descendants(widget):
            for child in widget.winfo_children():
                yield child
                yield from descendants(child)
        buttons = {w.cget("text"): w for w in descendants(self.root) if isinstance(w, tk.Button)}
        custom, new = buttons["套用自訂"], buttons["新的一局  F2"]
        self.assertEqual(custom.master, new.master)
        self.assertEqual(custom.winfo_rooty(), new.winfo_rooty())
        self.assertGreater(new.winfo_rootx(), custom.winfo_rootx() + custom.winfo_width())
        self.assertGreater(new.winfo_height(), 40)

    def test_presets_stop_new_game_and_stale_ai_results(self):
        self.app._preset((9, 9, 10))
        self.assertEqual(len(self.app.items), 81)
        self.app._preset((20, 30, 130))
        self.assertEqual(len(self.app.items), 600)
        self.app._preset((9, 9, 10))
        old_token, old_revision = self.app.token, self.app.session.revision
        self.app.stop_ai()
        action = LiveAction("reveal", (0, 0), "test", 0.0, True)
        self.app._accept_result(old_token, old_revision, action, None)
        self.assertEqual(self.app.session.clicks.total, 0)
        current_token = self.app.token
        self.app._act("flag", (1, 1))
        self.app._accept_result(current_token, old_revision, action, None)
        self.assertEqual(self.app.session.clicks.total, 1)
        self.assertEqual(self.app.session.game.status, GameStatus.READY)
        self.app._resize_board()
        self.root.update()
        self.assertEqual(len(self.app.items), 81)

    def test_close_during_ai_calculation_does_not_retain_or_touch_tk(self):
        from minesweeper_ai.agent import Decision
        started, release = threading.Event(), threading.Event()
        class BlockingAgent:
            def choose_move(self, observation, mines, *, allow_guess):
                started.set()
                release.wait(timeout=5)
                return Decision("reveal", (1, 1), 0.0, "test", True)
        with patch("minesweeper_ai.replay_deployment.load_replay_deployment",
                   return_value=(BlockingAgent(), None)):
            self.app.start_ai()
            deadline = time.monotonic() + 3
            while not started.is_set() and time.monotonic() < deadline:
                self.root.update()
                time.sleep(0.01)
            self.assertTrue(started.is_set())
            self.app.close()
            release.set()
            self.app.worker.join(timeout=2)
            self.assertFalse(self.app.worker.is_alive())
            self.assertEqual(self.app.session.clicks.total, 0)
            self.assertFalse(self.app.tile_refs)
            self.app.close()  # Closing again is safe.

    def test_losing_guess_retains_risk_and_saves_once(self):
        self.board(2, 2, ((0, 0),))
        self.app._act("reveal", (1, 1))
        action = LiveAction("reveal", (0, 0), "minimum risk", 1 / 3, True)
        self.app._accept_result(self.app.token, self.app.session.revision, action, None)
        self.assertEqual(self.app.session.game.status, GameStatus.LOST)
        self.assertIn("33.33%", self.app.ai_status.get())
        self.assertIn("紀錄已保存", self.app.ai_status.get())
        self.assertTrue(self.app.feedback_path.is_file())
        self.app._act("reveal", (0, 1))
        self.assertEqual(len(list(self.app.feedback_directory.glob("*.json"))), 1)

    def test_feedback_write_failure_keeps_game_usable(self):
        self.board(2, 2, ((0, 0),))
        self.app._act("reveal", (1, 1))
        with patch("minesweeper_ai.local_app.save_feedback", side_effect=OSError("disk test")):
            self.app._act("reveal", (0, 0))
        self.assertEqual(self.app.session.game.status, GameStatus.LOST)
        self.assertIn("紀錄儲存失敗", self.app.ai_status.get())
        self.app.new_game()
        self.assertEqual(self.app.session.game.status, GameStatus.READY)

    def test_actual_v44_completes_and_export(self):
        self.app.seed_input.set("20260905")
        self.app._preset((9, 9, 10))
        self.app.delay.set(0)
        self.app.start_ai()
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline and (self.app.running or self.app.busy):
            self.root.update()
            time.sleep(0.01)
        self.assertFalse(self.app.running, self.app.ai_status.get())
        self.assertEqual(self.app.session.game.status, GameStatus.WON, self.app.ai_status.get())
        self.assertEqual(self.app.session.three_bv, self.app.session.completed_bv)
        self.assertTrue(all(x["source"] == "V44" for x in self.app.session.history))
        Path("tmp").mkdir(exist_ok=True)
        with patch("minesweeper_ai.local_app.filedialog.asksaveasfilename",
                   return_value="tmp/local-gui-result.json"):
            self.app.export()
        result = json.loads(Path("tmp/local-gui-result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "won")
        self.assertTrue(self.app.feedback_path.is_file())
        self.assertEqual(result["training_status"], "recorded_not_trained")
        self.assertTrue(all(move["decision"] is not None for move in result["actions"]))
        self.assertEqual(len(result["actions"]), result["total_clicks"])
        print("V44_GUI_RESULT", json.dumps(self.app.session.summary()))
        self.root.update()
        x, y = self.app.canvas.winfo_rootx(), self.app.canvas.winfo_rooty()
        size = self.app.cell_pixels
        crop = ImageGrab.grab(bbox=(x, y, x+9*size, y+9*size), all_screens=True)
        read = HdSkinRecognizer(theme="dark").read_image(crop, 9, 9, total_mines=10)
        game = self.app.session.game
        for r, line in enumerate(read.observation):
            for c, value in enumerate(line):
                self.assertEqual(value, FLAGGED if game.is_mine((r,c)) else game.observation[r][c])
        self.assertEqual(read.terminal, "won")
        self.capture("tmp/local-gui-v44-win.png")

    def test_rapid_left_clicks_accept_press_even_with_release_drift(self):
        mines=[(r,c) for r in range(20) for c in range(30) if (r+c)%2==0]
        self.board(30,20,mines)
        targets=[(r,c) for r in range(20) for c in range(30) if (r+c)%2==1][:200]
        size=self.app.cell_pixels
        start=time.perf_counter()
        for r,c in targets:
            self.app.canvas.event_generate('<ButtonPress-1>',x=c*size+21,y=r*size+21,when='tail')
            self.app.canvas.event_generate('<ButtonRelease-1>',x=((c+1)%30)*size+21,y=r*size+21,when='tail')
        self.root.update()
        self.assertEqual(self.app.session.clicks.total,200)
        self.assertEqual(self.app.session.clicks.effective['left'],200)
        self.assertTrue(all(self.app.session.game.observation[r][c]>=0 for r,c in targets))
        self.assertEqual(self.app.session.game.status,GameStatus.ACTIVE)
        print('QUEUED_200_LEFT_CLICKS_SECONDS',round(time.perf_counter()-start,3))

    def test_first_press_survives_focus_out_and_does_not_chord_on_release(self):
        self.board()
        self.event('<ButtonPress-1>',(1,1))
        self.assertEqual(self.app.session.clicks.total,1)
        self.app.canvas.event_generate('<FocusOut>'); self.root.update()
        self.event('<ButtonRelease-1>',(1,2))
        self.assertEqual(self.app.session.clicks.total,1)
        self.assertGreaterEqual(self.app.session.game.observation[1][1],0)

    def test_replay_slider_playback_and_return_preserve_live_game(self):
        self.board()
        self.app._act('reveal',(1,1)); self.app._act('flag',(0,0))
        original=self.app.session; before=original.game.observation
        self.app.replay_current(); self.root.update()
        self.assertIs(self.app.session,original)
        self.assertIsNotNone(original.paused_at)
        self.assertEqual(self.app.replay_index,0)
        self.assertTrue(all(v==-1 for line in self.app.view_game.observation for v in line))
        self.app.replay_scale.set(2); self.root.update()
        self.assertEqual(self.app.replay_index,2)
        self.assertEqual(self.app.view_game.observation,before)
        self.app.start_ai(); self.assertFalse(self.app.running)
        self.event('<ButtonPress-1>',(2,3)); self.event('<ButtonRelease-1>',(2,3))
        self.assertEqual(original.clicks.total,2)
        self.app.seek_replay(0); self.app.replay_delay.set('0.1'); self.app.toggle_replay()
        deadline=time.monotonic()+2
        while self.app.replay_after is not None and time.monotonic()<deadline:
            self.root.update(); time.sleep(.01)
        self.assertEqual(self.app.replay_index,2)
        self.assertIsNone(self.app.replay_after)
        self.assertEqual(len(list(self.app.feedback_directory.glob('*.json'))),0)
        self.app.leave_replay(); self.root.update()
        self.assertIsNone(original.paused_at)
        self.assertIs(self.app.session,original)
        self.assertEqual(original.game.observation,before)
        self.assertEqual(original.clicks.total,2)

    def test_import_different_size_replay_and_evil_layout(self):
        from minesweeper_ai.local_replay import ReplayTimeline
        small=LocalGameSession(MinesweeperGame(2,2,1,mine_positions={(0,0)}))
        small.apply('reveal',(1,1)); small.apply('reveal',(0,0))
        timeline=ReplayTimeline.from_session(small,seed=1)
        self.app.enter_replay(timeline); self.root.update()
        self.app.seek_replay(2)
        self.assertEqual(len(self.app.items),4)
        self.assertEqual(self.app.view_game.status,GameStatus.LOST)
        self.app.seek_replay(1)
        self.assertNotIn(('exploded',False),self.app.displayed.values())
        self.app.leave_replay(); self.root.update()
        self.assertEqual(len(self.app.items),600)
        self.app._act('reveal',(10,15)); self.app.replay_current(); self.root.update()
        self.app.seek_replay(1)
        self.assertGreaterEqual(self.app.canvas.winfo_width(),30*42)
        self.assertGreaterEqual(self.app.canvas.winfo_height(),20*42)
        Path('tmp').mkdir(exist_ok=True)
        self.capture('tmp/local-gui-replay-evil.png')
        self.app.new_game(); self.assertIsNone(self.app.replay)

    def test_optional_marker_defaults_off_and_tracks_live_and_replay(self):
        self.board()
        self.assertFalse(self.app.show_action_marker.get())
        self.app._act('reveal',(1,1))
        self.app._act('flag',(0,0))
        self.assertFalse(self.app.canvas.find_withtag('action-marker'))
        self.app.marker_check.invoke()
        self.assertEqual(len(self.app.canvas.find_withtag('action-marker')),1)
        self.assertEqual(self.app.canvas.coords('action-marker'),[1.,1.,41.,41.])
        self.app.replay_current(); self.root.update()
        self.assertFalse(self.app.canvas.find_withtag('action-marker'))
        self.app.seek_replay(1)
        self.assertEqual(self.app.canvas.coords('action-marker'),[43.,43.,83.,83.])
        self.app.marker_check.invoke()
        self.assertFalse(self.app.canvas.find_withtag('action-marker'))
        self.app.seek_replay(2)
        self.assertFalse(self.app.canvas.find_withtag('action-marker'))
        self.app.leave_replay()
        self.assertFalse(self.app.canvas.find_withtag('action-marker'))
        self.assertEqual(self.app.session.clicks.total,2)

    def test_win_auto_flags_unflagged_mines_and_rewind_removes_completion_flags(self):
        self.board()
        self.app._act('flag',(0,0))
        for r in range(4):
            for c in range(4):
                if not self.app.session.game.is_mine((r,c)):
                    self.app._act('reveal',(r,c))
        self.assertEqual(self.app.session.game.status,GameStatus.WON)
        before=self.app.session.game.observation
        clicks=self.app.session.clicks.total
        self.assertEqual(self.app.session.game.remaining_mines,1)
        self.assertIn('剩餘雷 0',self.app.board_info.get())
        self.assertEqual(self.app.displayed[(0,0)],(FLAGGED,False))
        self.assertEqual(self.app.displayed[(0,3)],(FLAGGED,False))
        self.app.replay_current(); self.root.update()
        self.assertIn('剩餘雷 2',self.app.board_info.get())
        self.assertEqual(self.app.displayed[(0,3)],(-1,False))
        self.app.seek_replay(clicks)
        self.assertIn('剩餘雷 0',self.app.board_info.get())
        self.assertEqual(self.app.displayed[(0,3)],(FLAGGED,False))
        self.assertEqual(self.app.displayed[(0,0)],(FLAGGED,False))
        self.app.seek_replay(clicks-1)
        self.assertIn('剩餘雷 1',self.app.board_info.get())
        self.assertEqual(self.app.displayed[(0,3)],(-1,False))
        self.app.leave_replay()
        self.assertEqual(self.app.session.game.observation,before)
        self.assertEqual(self.app.session.clicks.total,clicks)

    def test_loss_preserves_flags_reveals_mines_and_zeroes_display_counter(self):
        self.board(4,4,((0,0),(0,3),(3,3)))
        self.app._act('flag',(0,0))
        self.app._act('reveal',(1,1))
        self.app._act('reveal',(0,3))
        self.assertEqual(self.app.session.game.status,GameStatus.LOST)
        self.assertIn('剩餘雷 0',self.app.board_info.get())
        self.assertEqual(self.app.displayed[(0,0)],(FLAGGED,False))
        self.assertEqual(self.app.displayed[(0,3)],('exploded',False))
        self.assertEqual(self.app.displayed[(3,3)],('mine',False))
        self.assertEqual(self.app.session.clicks.total,3)

    def test_replay_analysis_background_cache_and_stale_result(self):
        self.board(2,2,((0,0),))
        self.app._act('reveal',(1,1)); self.app._act('reveal',(0,0))
        self.app.replay_current(); self.app.seek_replay(2)
        self.app.analysis_button.invoke()
        deadline=time.monotonic()+5
        while self.app.analysis_pending and time.monotonic()<deadline:
            self.root.update(); time.sleep(.01)
        self.assertIsNone(self.app.analysis_pending)
        self.assertIn('33.33%',self.app.analysis_text.get())
        self.assertIn('實際展開 0 個安全格',self.app.analysis_text.get())
        with patch.object(self.app.requests,'put',side_effect=AssertionError('cached step recomputed')):
            self.app.seek_replay(1); self.app.seek_replay(2); self.app.analysis_button.invoke()
        old_epoch=self.app.analysis_epoch
        self.app.analysis_pending=(old_epoch,1)
        self.app.leave_replay(); self.app.replay_current(); self.app.seek_replay(1)
        self.app._accept_analysis(dict(key=(old_epoch,1),text='STALE'))
        self.assertNotIn('STALE',self.app.analysis_text.get())
        self.assertEqual(self.app.analysis_cache,{})
        self.app.seek_replay(2); self.app.analysis_button.invoke()
        deadline=time.monotonic()+5
        while self.app.analysis_pending and time.monotonic()<deadline:
            self.root.update(); time.sleep(.01)
        self.capture('tmp/local-replay-analysis.png')

    def capture(self, path):
        self.root.update()
        time.sleep(0.18)
        self.root.update()
        x, y = self.root.winfo_rootx(), self.root.winfo_rooty()
        ImageGrab.grab(bbox=(x, y, x+self.root.winfo_width(), y+self.root.winfo_height()),
                       all_screens=True).save(path)

    def test_layout_and_preview_screenshots(self):
        self.app.seed_input.set("20260905")
        self.app._preset((16, 30, 99))
        self.app._act("reveal", (8, 15))
        obs = self.app.session.game.observation
        number = next((r, c) for r in range(16) for c in range(30)
                      if obs[r][c] > 0 and preview_cells(obs, (r, c)))
        flag = next(iter(preview_cells(obs, number)))
        self.app._act("flag", flag)
        self.root.update()
        Path("tmp").mkdir(exist_ok=True)
        self.capture("tmp/local-gui-normal.png")
        self.event("<ButtonPress-1>", number)
        self.assertNotIn(flag, self.app.shaded)
        self.capture("tmp/local-gui-hold.png")
        self.app._clear_press()
        self.app._preset((20, 30, 130))
        self.root.update()
        self.capture("tmp/local-gui-evil.png")

if __name__ == "__main__":
    unittest.main()
