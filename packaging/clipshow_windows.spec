# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for ClipShow Windows folder-mode build."""

import os

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

# Semantic detector lazy imports
semantic_hiddenimports = [
    "onnx_clip",
    "onnxruntime",
    "onnxruntime.capi",
    "onnxruntime.capi.onnxruntime_pybind11_state",
    "PIL",
    "PIL.Image",
    "scipy",
    "scipy.ndimage",
]

# Collect onnx_clip model data only for "full" builds
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, copy_metadata

bundle_models = os.environ.get("CLIPSHOW_BUNDLE_MODELS", "0") == "1"
onnx_clip_datas = collect_data_files("onnx_clip") if bundle_models else []
onnxruntime_binaries = collect_dynamic_libs("onnxruntime")

# Include pre-downloaded models for "full" builds
bundled_model_datas = []
if bundle_models:
    models_dir = os.path.join(os.path.dirname(os.path.abspath(SPECPATH)), "bundled_models")
    if os.path.isdir(models_dir):
        for f in os.listdir(models_dir):
            bundled_model_datas.append(
                (os.path.join(models_dir, f), os.path.join("models", f))
            )

# imageio checks its own version via importlib.metadata at import time
imageio_metadata = copy_metadata("imageio")

a = Analysis(
    ["../clipshow/__main__.py"],
    pathex=[],
    binaries=[*onnxruntime_binaries],
    datas=[*onnx_clip_datas, *bundled_model_datas, *imageio_metadata],
    hiddenimports=[
        *scenedetect_hiddenimports,
        *pyside6_hiddenimports,
        *semantic_hiddenimports,
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
    icon=None,
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
