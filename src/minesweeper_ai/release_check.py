"""A small check inside the actual binary; no training or large benchmark."""
import json
from pathlib import Path
import tempfile
import time
import traceback
def run(output):
    result={'passed':False}
    app=None
    try:
        import torch
        import tkinter as tk
        from .game import GameStatus
        from .local_app import LocalMinesweeperApp, MODEL_CONFIG
        from .runtime_paths import feedback_directory
        assert MODEL_CONFIG.is_file(), 'Missing bundled V44'
        with tempfile.TemporaryDirectory() as folder:
            root=tk.Tk(); root.withdraw()
            errors=[]
            root.report_callback_exception=lambda *args:errors.append(str(args))
            app=LocalMinesweeperApp(root,feedback_directory=Path(folder))
            app.seed_input.set('20260905')
            app._preset((9,9,10))
            app._act('flag',(0,0)); app._act('flag',(0,0))
            assert app.session.clicks.total==2
            app.new_game(); app.delay.set(0); app.start_ai()
            deadline=time.monotonic()+60
            while time.monotonic()<deadline and (app.running or app.busy):
                root.update(); time.sleep(.01)
            assert app.session.game.status==GameStatus.WON, app.ai_status.get()
            assert app.feedback_path and app.feedback_path.is_file()
            saved=json.loads(app.feedback_path.read_text(encoding='utf-8'))
            assert saved['training_status']=='recorded_not_trained'
            original=app.session
            app.replay_current(); app.seek_replay(2); app.analysis_button.invoke()
            deadline=time.monotonic()+30
            while app.analysis_pending and time.monotonic()<deadline:
                root.update(); time.sleep(.01)
            assert app.analysis_pending is None
            assert app.analysis_cache and '分析失敗' not in app.analysis_text.get()
            app.seek_replay(len(original.history))
            assert app.view_game.status==GameStatus.WON
            app.leave_replay()
            assert app.session is original
            assert not errors, errors
            result.update(torch=torch.__version__,ai_won=True,actions=len(original.history),manual_flags=True,replay_analysis=True,saved_feedback=True,returned_to_original_game=True,per_user_storage=feedback_directory().parts[-2:]==('MinesweeperAI','local-feedback'))
            app.close(); app.worker.join(timeout=5)
            assert not app.worker.is_alive()
            app=None
            result['passed']=True
    except Exception:
        result['error']=traceback.format_exc()
    finally:
        if app is not None and not app.closed:app.close()
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    if not result['passed']:raise SystemExit(1)
