# PyInstaller spec — builds the one-file VoynichWorkbench executable.
# Run from the repository root:  pyinstaller voynich_workbench.spec
import os

from PyInstaller.utils.hooks import collect_data_files

datas = [
    (os.path.join("voynich", "webapp", "templates"),
     os.path.join("voynich", "webapp", "templates")),
    (os.path.join("voynich", "webapp", "static"),
     os.path.join("voynich", "webapp", "static")),
]
# pypinyin ships its pinyin dictionaries as package data.
datas += collect_data_files("pypinyin")

a = Analysis(
    ["desktop.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=["voynich.webapp.app"],
    hookspath=[],
    runtime_hooks=[],
    excludes=["matplotlib", "tkinter"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="VoynichWorkbench",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    icon=None,
)
