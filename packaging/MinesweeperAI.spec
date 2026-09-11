from pathlib import Path
root=Path(SPECPATH).parent
datas=[(str(root/'models/replay-agent-v44.json'),'models'),(str(root/'models/replay-agent-v40-calibrated-abstention.pt'),'models')]
a=Analysis([str(root/'packaging/launch_local.py')],pathex=[str(root/'src')],binaries=[],datas=datas,hiddenimports=[],hookspath=[],hooksconfig={},runtime_hooks=[],excludes=['matplotlib','scipy','pytest','IPython','notebook','tensorboard'],noarchive=False)
pyz=PYZ(a.pure)
exe=EXE(pyz,a.scripts,[],exclude_binaries=True,name='MinesweeperAI',debug=False,bootloader_ignore_signals=False,strip=False,upx=False,console=False,disable_windowed_traceback=False)
coll=COLLECT(exe,a.binaries,a.datas,strip=False,upx=False,name='MinesweeperAI')
