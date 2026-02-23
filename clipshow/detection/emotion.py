"""Face/emotion detection via MediaPipe + deepface-onnx (no TensorFlow).

Lazy-loads mediapipe and deepface_onnx on first use. Samples at 3 FPS.
Scores frames higher when faces show positive/high-energy emotions
(happy, surprise).
"""

from __future__ import annotations

import importlib

import cv2
import numpy as np

from clipshow.detection.base import Detector, DetectorResult

SAMPLE_FPS = 3
POSITIVE_EMOTIONS = {"happy", "surprise"}
NEUTRAL_SCORE = 0.2  # Score for neutral faces (still somewhat interesting)


class EmotionDetector(Detector):
    """Face detection + emotion scoring.

    Uses MediaPipe for face detection and deepface-onnx for emotion
    classification. Returns higher scores for positive/high-energy emotions.
    """

    name = "emotion"

    def __init__(self, time_step: float = 0.1):
        self._time_step = time_step
        self._face_detector = None
        self._emotion_model = None

    def _load_models(self):
        """Lazy-load face detection and emotion models."""
        try:
            mp = importlib.import_module("mediapipe")
        except ImportError:
            raise RuntimeError(
                "mediapipe is not installed. Install with: pip install mediapipe"
            )

        try:
            deepface_onnx = importlib.import_module("deepface_onnx")
        except ImportError:
            raise RuntimeError(
                "deepface-onnx is not installed. Install with: pip install deepface-onnx"
            )

        self._face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5,
        )
        self._emotion_model = deepface_onnx

    def detect(
        self,
        video_path: str,
        progress_callback: callable | None = None,
        cancel_flag: callable | None = None,
    ) -> DetectorResult:
        if self._face_detector is None:
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
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_detector.process(rgb)

        if not results.detections:
            return 0.0

        max_score = 0.0
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w = frame.shape[:2]
            x1 = max(0, int(bbox.xmin * w))
            y1 = max(0, int(bbox.ymin * h))
            x2 = min(w, int((bbox.xmin + bbox.width) * w))
            y2 = min(h, int((bbox.ymin + bbox.height) * h))

            if x2 <= x1 or y2 <= y1:
                continue

            face_crop = frame[y1:y2, x1:x2]
            try:
                emotions = self._emotion_model.analyze(face_crop)
                dominant = emotions.get("dominant_emotion", "")
                if dominant in POSITIVE_EMOTIONS:
                    max_score = max(max_score, 1.0)
                elif dominant == "neutral":
                    max_score = max(max_score, NEUTRAL_SCORE)
                else:
                    max_score = max(max_score, 0.1)
            except Exception:
                # Face crop too small or model error — skip
                max_score = max(max_score, 0.1)

        return max_score
