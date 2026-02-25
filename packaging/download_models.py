"""Download ONNX models for bundling in full release builds.

Downloads models to packaging/bundled_models/ where the PyInstaller specs
pick them up. Also downloads CLIP models into the onnx_clip package data
directory so collect_data_files() includes them.
"""

import urllib.request
from pathlib import Path

MODELS = {
    "emotion-ferplus-8.onnx": (
        "https://github.com/onnx/models/raw/main/validated/vision/"
        "body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
    ),
}

CLIP_MODELS = {
    "clip_image_model_vitb32.onnx": (
        "https://lakera-clip.s3.eu-west-1.amazonaws.com/clip_image_model_vitb32.onnx"
    ),
    "clip_text_model_vitb32.onnx": (
        "https://lakera-clip.s3.eu-west-1.amazonaws.com/clip_text_model_vitb32.onnx"
    ),
}


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  Already exists: {dest} ({dest.stat().st_size / 1024 / 1024:.1f}MB)")
        return
    print(f"  Downloading: {dest.name} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    print(f"  Done: {dest.stat().st_size / 1024 / 1024:.1f}MB")


def main():
    script_dir = Path(__file__).resolve().parent
    bundled_dir = script_dir / "bundled_models"

    print("Downloading models for full build...")

    # 1. Emotion model -> bundled_models/
    for filename, url in MODELS.items():
        download(url, bundled_dir / filename)

    # 2. CLIP models -> onnx_clip package data directory
    #    (so collect_data_files('onnx_clip') includes them)
    try:
        import onnx_clip
        onnx_clip_data = Path(onnx_clip.__file__).parent / "data"
    except ImportError:
        print("WARNING: onnx_clip not installed, skipping CLIP model download")
        return

    for filename, url in CLIP_MODELS.items():
        download(url, onnx_clip_data / filename)

    print("\nAll models downloaded successfully.")
    total = sum(
        f.stat().st_size
        for d in [bundled_dir, onnx_clip_data]
        if d.exists()
        for f in d.iterdir()
        if f.suffix == ".onnx"
    )
    print(f"Total ONNX model size: {total / 1024 / 1024:.0f}MB")


if __name__ == "__main__":
    main()
