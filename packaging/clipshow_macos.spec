# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for ClipShow macOS .app bundle."""

import sys
from pathlib import Path

block_cipher = None

# Collect scenedetect submodules which use lazy imports
scenedetect_hiddenimports = [
    "scenedetect",
    "scenedetect.detectors",
    "scenedetect.detectors.content_detector",
    "scenedetect.detectors.threshold_detector",
    "scenedetect.detectors.adaptive_detector",
    "scenedetect.scene_manager",
    "scenedetect.video_splitter",
    "scenedetect.stats_manager",
    "scenedetect.frame_timecode",
]

pyside6_hiddenimports = [
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets",
    "PySide6.QtNetwork",
]

a = Analysis(
    ["../clipshow/__main__.py"],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        *scenedetect_hiddenimports,
        *pyside6_hiddenimports,
        "cv2",
        "numpy",
        "moviepy",
        "librosa",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib",
        "IPython",
        "jupyter",
        "notebook",
        "pytest",
        "sphinx",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ClipShow",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    target_arch="universal2",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="ClipShow",
)

app = BUNDLE(
    coll,
    name="ClipShow.app",
    icon=None,
    bundle_identifier="com.clipshow.app",
    info_plist={
        "CFBundleName": "ClipShow",
        "CFBundleDisplayName": "ClipShow",
        "CFBundleShortVersionString": "0.1.0",
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": "11.0",
    },
)
