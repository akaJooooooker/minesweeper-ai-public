"""Bundled models and writable per-user storage."""
import os
from pathlib import Path
import sys
def resource_root():
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parents[2]
def feedback_directory():
    if getattr(sys, 'frozen', False):
        base=Path(os.environ.get('LOCALAPPDATA', Path.home()/'AppData'/'Local'))
        return base/'MinesweeperAI'/'local-feedback'
    return resource_root()/'data'/'local-feedback'
