# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for RealtimeTranslator (supports CPU and CUDA builds via RTT_BUILD_CUDA=1)."""

import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_data_files

datas = []
binaries = []
hiddenimports = []

for pkg in ("faster_whisper", "ctranslate2", "onnxruntime"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

datas += collect_data_files("PySide6")

if os.environ.get("RTT_BUILD_CUDA") == "1":
    d_nv, b_nv, h_nv = collect_all("nvidia")
    datas += d_nv
    binaries += b_nv
    hiddenimports += h_nv

local_silero = Path("models") / "silero_vad.onnx"
appdata_silero = Path(os.environ.get("APPDATA", "")) / "RealtimeTranslator" / "models" / "silero_vad.onnx"
if local_silero.is_file():
    datas.append((str(local_silero), "models"))
elif appdata_silero.is_file():
    datas.append((str(appdata_silero), "models"))

a = Analysis(
    ["rtt\\__main__.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["torch"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="RealtimeTranslator",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="RealtimeTranslator",
)
