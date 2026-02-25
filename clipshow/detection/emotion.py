"""Face/emotion detection via OpenCV + emotion-ferplus ONNX model.

Uses OpenCV's Haar cascade for face detection and a small ONNX model
(emotion-ferplus-8, ~35KB) for emotion classification. Downloads model
to ~/.clipshow/models/ on first use. Samples at 3 FPS.
Scores frames higher when faces show positive/high-energy emotions
(happy, surprise).
"""

from __future__ import annotations

import importlib
import logging
import tempfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from clipshow.detection.base import Detector, DetectorResult

logger = logging.getLogger(__name__)

SAMPLE_FPS = 3
POSITIVE_EMOTIONS = {"happiness", "surprise"}
NEUTRAL_SCORE = 0.2  # Score for neutral faces (still somewhat interesting)
MODEL_DIR = Path.home() / ".clipshow" / "models"
MODEL_FILENAME = "emotion-ferplus-8.onnx"
MODEL_URL = (
    "https://github.com/onnx/models/raw/main/validated/vision/"
    "body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
)

# emotion-ferplus output labels in order
EMOTION_LABELS = [
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt",
]


class EmotionDetector(Detector):
    """Face detection + emotion scoring.

    Uses OpenCV Haar cascade for face detection and emotion-ferplus ONNX
    model for emotion classification. Returns higher scores for
    positive/high-energy emotions.
    """

    name = "emotion"

    def __init__(self, time_step: float = 0.1):
        self._time_step = time_step
        self._face_cascade = None
        self._emotion_session = None

    def _load_models(self):
        """Lazy-load face detection cascade and emotion ONNX model."""
        try:
            ort = importlib.import_module("onnxruntime")
        except ImportError:
            raise RuntimeError(
                "onnxruntime is not installed. Install with: "
                "uv sync --extra emotion"
            )

        # OpenCV Haar cascade for face detection (bundled with opencv-python-headless)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._face_cascade = cv2.CascadeClassifier(cascade_path)

        # Download emotion-ferplus ONNX model if needed (atomic to avoid races)
        model_path = MODEL_DIR / MODEL_FILENAME
        if not model_path.exists():
            logger.info("Downloading emotion-ferplus model to %s", model_path)
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(dir=MODEL_DIR, suffix=".tmp")
            try:
                import os

                os.close(fd)
                urllib.request.urlretrieve(MODEL_URL, tmp_path)
                os.replace(tmp_path, model_path)
            except BaseException:
                # Clean up partial download on any failure
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

        self._emotion_session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )

    def detect(
        self,
        video_path: str,
        progress_callback: callable | None = None,
        cancel_flag: callable | None = None,
    ) -> DetectorResult:
        if self._face_cascade is None:
            self._load_models()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0.0
        frame_interval = max(1, int(fps / SAMPLE_FPS))

        num_samples = max(1, int(np.ceil(duration / self._time_step)))
        scores = np.zeros(num_samples, dtype=float)

        frame_idx = 0
        while True:
            if cancel_flag and cancel_flag():
                break

            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                score = self._score_frame(frame)
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

    def _score_frame(self, frame: np.ndarray) -> float:
        """Score a single frame for face/emotion content."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        if len(faces) == 0:
            return 0.0

        max_score = 0.0
        for x, y, w, h in faces:
            face_crop = gray[y : y + h, x : x + w]
            if face_crop.size == 0:
                continue

            try:
                dominant = self._classify_emotion(face_crop)
                if dominant in POSITIVE_EMOTIONS:
                    max_score = max(max_score, 1.0)
                elif dominant == "neutral":
                    max_score = max(max_score, NEUTRAL_SCORE)
                else:
                    max_score = max(max_score, 0.1)
            except Exception:
                # Face crop too small or model error -- skip
                max_score = max(max_score, 0.1)

        return max_score

    def _classify_emotion(self, gray_face: np.ndarray) -> str:
        """Run emotion-ferplus ONNX model on a grayscale face crop."""
        # emotion-ferplus expects 1x1x64x64 float32 input
        resized = cv2.resize(gray_face, (64, 64)).astype(np.float32)
        tensor = resized.reshape(1, 1, 64, 64)

        input_name = self._emotion_session.get_inputs()[0].name
        outputs = self._emotion_session.run(None, {input_name: tensor})

        # Softmax over logits to get probabilities
        logits = outputs[0][0]
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        return EMOTION_LABELS[int(np.argmax(probs))]
