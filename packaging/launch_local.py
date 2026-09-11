"""Windowed release entry point."""
import multiprocessing
import sys
import traceback
from pathlib import Path
if __name__ == '__main__':
    multiprocessing.freeze_support()
    if '--self-test' in sys.argv:
        from minesweeper_ai.release_check import run
        run(Path(sys.argv[sys.argv.index('--self-test')+1]))
    else:
        try:
            from minesweeper_ai.local_app import main
            main()
        except Exception:
            from minesweeper_ai.runtime_paths import feedback_directory
            log=feedback_directory().parent/'startup-error.log'
            log.parent.mkdir(parents=True,exist_ok=True)
            log.write_text(traceback.format_exc(),encoding='utf-8')
            import tkinter as tk
            from tkinter import messagebox
            root=tk.Tk(); root.withdraw()
            messagebox.showerror('啟動失敗', f'錯誤紀錄已儲存至：{log}')
            root.destroy()
            raise SystemExit(1)
