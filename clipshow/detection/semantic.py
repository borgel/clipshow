"""Semantic content detection via CLIP (ONNX Runtime, no PyTorch).

Lazy-loads onnx_clip on first use. Downloads CLIP ViT-B/32 model (~338MB)
to ~/.clipshow/models/ on first use. Samples frames at 1 FPS and scores
against user-configurable text prompts.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import cv2
import numpy as np

from clipshow.detection.base import Detector, DetectorResult

DEFAULT_PROMPTS = [
    "exciting moment",
    "people laughing",
    "beautiful scenery",
    "action scene",
]
MODEL_DIR = Path.home() / ".clipshow" / "models"
SAMPLE_FPS = 1  # Sample 1 frame per second


class SemanticDetector(Detector):
    """CLIP-based semantic content scoring.

    Uses onnx_clip (ONNX Runtime) to score video frames against text prompts.
    Returns cosine similarity as score, normalized to [0, 1].
    """

    name = "semantic"

    def __init__(
        self,
        prompts: list[str] | None = None,
        time_step: float = 0.1,
    ):
        self._prompts = prompts or DEFAULT_PROMPTS
        self._time_step = time_step
        self._model = None

    def _load_model(self):
        """Lazy-load the CLIP model via onnx_clip."""
        try:
            onnx_clip = importlib.import_module("onnx_clip")
        except ImportError:
            raise RuntimeError(
                "onnx_clip is not installed. Install with: "
                "pip install onnx_clip onnxruntime"
            )
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self._model = onnx_clip.OnnxClip(batch_size=1)
        return self._model

    def detect(
        self,
        video_path: str,
        progress_callback: callable | None = None,
        cancel_flag: callable | None = None,
    ) -> DetectorResult:
        model = self._model or self._load_model()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0.0
        frame_interval = max(1, int(fps / SAMPLE_FPS))

        num_samples = max(1, int(np.ceil(duration / self._time_step)))
        scores = np.zeros(num_samples, dtype=float)

        # Encode text prompts once
        from PIL import Image

        text_embeddings = model.get_text_embeddings(self._prompts)

        frame_idx = 0
        while True:
            if cancel_flag and cancel_flag():
                break

            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                # Convert BGR to RGB PIL Image
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)

                image_embeddings = model.get_image_embeddings([img])

                # Cosine similarity against all prompts, take max
                similarities = image_embeddings @ text_embeddings.T
                score = float(np.max(similarities))
                # CLIP similarities are typically in [-1, 1], normalize to [0, 1]
                score = max(0.0, min(1.0, (score + 1.0) / 2.0))

                t = frame_idx / fps
                idx = min(int(t / self._time_step), num_samples - 1)
                scores[idx] = max(scores[idx], score)

                if progress_callback:
                    progress_callback(frame_idx / max(total_frames, 1))

            frame_idx += 1

        cap.release()

        # Normalize to [0, 1]
        max_val = scores.max()
        if max_val > 0:
            scores = scores / max_val

        if progress_callback:
            progress_callback(1.0)

        return DetectorResult(
            name=self.name,
            scores=scores,
            time_step=self._time_step,
            source_path=video_path,
        )
