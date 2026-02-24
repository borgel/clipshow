"""ONNX model download, cache, and load infrastructure.

Provides ModelManager for centralized model management across all
detectors that use ONNX Runtime (semantic, emotion, audio-visual).
Downloads models on first use, caches in ~/.clipshow/models/, and
selects the best available execution provider (CUDA > CoreML > CPU).
"""

from __future__ import annotations

from pathlib import Path

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
    """Download, cache, and load ONNX models.

    Stub — tests define the interface (TDD red phase).
    Implementation in clipshow-hph will make tests pass (green).
    """

    cache_dir: Path = Path.home() / ".clipshow" / "models"

    def __init__(self, cache_dir: Path | None = None):
        if cache_dir is not None:
            self.cache_dir = cache_dir

    def ensure_model(self, name: str, progress_cb=None) -> Path:
        """Download model if not cached, return local path."""
        raise NotImplementedError

    def load_session(self, name: str, **kwargs):
        """Load an ONNX Runtime session for a model."""
        raise NotImplementedError

    def _get_providers(self) -> list[str]:
        """Detect available execution providers, prefer GPU."""
        raise NotImplementedError
