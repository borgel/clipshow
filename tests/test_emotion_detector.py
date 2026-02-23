"""Tests for emotion detector with mocked mediapipe/deepface."""

import importlib
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from clipshow.detection.base import DetectorResult


class TestEmotionDetector:
    def test_missing_mediapipe_raises(self):
        """Should raise RuntimeError if mediapipe is not installed."""
        from clipshow.detection.emotion import EmotionDetector

        original_import = importlib.import_module

        def selective_import(name):
            if name == "mediapipe":
                raise ImportError("no mediapipe")
            return original_import(name)

        detector = EmotionDetector()
        with patch("importlib.import_module", side_effect=selective_import):
            with pytest.raises(RuntimeError, match="mediapipe is not installed"):
                detector._load_models()

    def test_missing_deepface_raises(self):
        """Should raise RuntimeError if deepface-onnx is not installed."""
        from clipshow.detection.emotion import EmotionDetector

        original_import = importlib.import_module

        def selective_import(name):
            if name == "mediapipe":
                return MagicMock()
            if name == "deepface_onnx":
                raise ImportError("no deepface_onnx")
            return original_import(name)

        detector = EmotionDetector()
        with patch("importlib.import_module", side_effect=selective_import):
            with pytest.raises(RuntimeError, match="deepface-onnx is not installed"):
                detector._load_models()

    def test_detect_no_faces_returns_zeros(self, static_video):
        """Video with no faces should produce zero scores."""
        from clipshow.detection.emotion import EmotionDetector

        mock_face = MagicMock()
        mock_face.process.return_value = MagicMock(detections=None)

        detector = EmotionDetector()
        detector._face_detector = mock_face
        detector._emotion_model = MagicMock()

        result = detector.detect(static_video)

        assert isinstance(result, DetectorResult)
        assert result.name == "emotion"
        assert result.scores.max() == 0.0

    def test_cancel_flag_stops_processing(self, static_video):
        """Cancel flag should stop frame processing."""
        from clipshow.detection.emotion import EmotionDetector

        mock_face = MagicMock()
        mock_face.process.return_value = MagicMock(detections=None)

        detector = EmotionDetector()
        detector._face_detector = mock_face
        detector._emotion_model = MagicMock()

        result = detector.detect(static_video, cancel_flag=lambda: True)
        assert isinstance(result, DetectorResult)

    def test_score_frame_no_faces(self):
        """_score_frame with no detections returns 0."""
        from clipshow.detection.emotion import EmotionDetector

        mock_face = MagicMock()
        mock_face.process.return_value = MagicMock(detections=None)

        detector = EmotionDetector()
        detector._face_detector = mock_face

        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        assert detector._score_frame(frame) == 0.0
