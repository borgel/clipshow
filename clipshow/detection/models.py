"""ONNX model download, cache, and load infrastructure.

Provides ModelManager for centralized model management across all
detectors that use ONNX Runtime (semantic, emotion, audio-visual).
Downloads models on first use, caches in ~/.clipshow/models/, and
selects the best available execution provider (CUDA > CoreML > CPU).
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_REGISTRY: dict[str, dict] = {
    # Phase 3: SigLIP upgrade
    "siglip-vit-b-16": {
        "file": "siglip-vit-b-16-image.onnx",
        "url": "https://github.com/clipshow/models/releases/download/v1/siglip-vit-b-16-image.onnx",
        "size_mb": 600,
    },
    # Phase 7: LanguageBind audio-visual
    "languagebind-video": {
        "file": "languagebind-video-encoder.onnx",
        "url": "https://github.com/clipshow/models/releases/download/v2/languagebind-video-encoder.onnx",
        "size_mb": 600,
    },
    "languagebind-audio": {
        "file": "languagebind-audio-encoder.onnx",
        "url": "https://github.com/clipshow/models/releases/download/v2/languagebind-audio-encoder.onnx",
        "size_mb": 400,
    },
    "languagebind-text": {
        "file": "languagebind-text-encoder.onnx",
        "url": "https://github.com/clipshow/models/releases/download/v2/languagebind-text-encoder.onnx",
        "size_mb": 500,
    },
}


class ModelManager:
    """Download, cache, and load ONNX models."""

    cache_dir: Path = Path.home() / ".clipshow" / "models"

    def __init__(self, cache_dir: Path | None = None):
        if cache_dir is not None:
            self.cache_dir = cache_dir

    def ensure_model(self, name: str, progress_cb=None) -> Path:
        """Download model if not cached, return local path."""
        if name not in MODEL_REGISTRY:
            raise KeyError(f"Unknown model: {name!r}")

        meta = MODEL_REGISTRY[name]
        model_path = self.cache_dir / meta["file"]

        # Skip download if file exists and is non-empty
        if model_path.exists() and model_path.stat().st_size > 0:
            return model_path

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Bridge urlretrieve's reporthook to user's progress_cb
        reporthook = None
        if progress_cb is not None:

            def reporthook(block_num, block_size, total_size):
                if total_size > 0:
                    progress = min(1.0, (block_num * block_size) / total_size)
                else:
                    progress = 0.0
                progress_cb(progress)

        logger.info("Downloading model %r to %s", name, model_path)
        urllib.request.urlretrieve(meta["url"], str(model_path), reporthook=reporthook)

        return model_path

    def load_session(self, name: str, **kwargs):
        """Load an ONNX Runtime session for a model."""
        import onnxruntime as ort

        path = self.ensure_model(name)
        providers = self._get_providers()
        return ort.InferenceSession(str(path), providers=providers)

    def _get_providers(self) -> list[str]:
        """Detect available execution providers, prefer GPU."""
        import onnxruntime as ort

        available = ort.get_available_providers()
        preferred = [
            "CUDAExecutionProvider",
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ]
        return [p for p in preferred if p in available]
