# Windows x64 build

Use 64-bit Python 3.12 with Tk from repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements/build.txt
.\.venv\Scripts\python.exe -m PyInstaller --distpath dist --workpath build packaging/MinesweeperAI.spec
```

Distribute all of `dist/MinesweeperAI/` with LICENSE, NOTICE.md and third-party runtime licenses. PyInstaller's bootloader exception and dependency licenses remain applicable.

Run `MinesweeperAI.exe --self-test C:\path\binary-check.json` from an unrelated working directory. The hidden check loads V44, plays one beginner game, saves feedback in a temporary directory, analyses a replay step, seeks and returns to the live game. No large benchmark.

Player records use `%LOCALAPPDATA%\MinesweeperAI\local-feedback`; installation folder need not be writable.
