"""Tests for emotion detector with mocked OpenCV/ONNX Runtime."""

import importlib
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from clipshow.detection.base import DetectorResult


class TestEmotionDetector:
    def test_missing_onnxruntime_raises(self):
        """Should raise RuntimeError if onnxruntime is not installed."""
        from clipshow.detection.emotion import EmotionDetector

        original_import = importlib.import_module

        def selective_import(name):
            if name == "onnxruntime":
                raise ImportError("no onnxruntime")
            return original_import(name)

        detector = EmotionDetector()
        with patch("importlib.import_module", side_effect=selective_import):
            with pytest.raises(RuntimeError, match="onnxruntime is not installed"):
                detector._load_models()

    def test_detect_no_faces_returns_zeros(self, static_video):
        """Video with no faces should produce zero scores."""
        from clipshow.detection.emotion import EmotionDetector

        mock_cascade = MagicMock()
        mock_cascade.detectMultiScale.return_value = ()

        detector = EmotionDetector()
        detector._face_cascade = mock_cascade
        detector._emotion_session = MagicMock()

        result = detector.detect(static_video)

        assert isinstance(result, DetectorResult)
        assert result.name == "emotion"
        assert result.scores.max() == 0.0

    def test_cancel_flag_stops_processing(self, static_video):
        """Cancel flag should stop frame processing."""
        from clipshow.detection.emotion import EmotionDetector

        mock_cascade = MagicMock()
        mock_cascade.detectMultiScale.return_value = ()

        detector = EmotionDetector()
        detector._face_cascade = mock_cascade
        detector._emotion_session = MagicMock()

        result = detector.detect(static_video, cancel_flag=lambda: True)
        assert isinstance(result, DetectorResult)

    def test_score_frame_no_faces(self):
        """_score_frame with no detections returns 0."""
        from clipshow.detection.emotion import EmotionDetector

        mock_cascade = MagicMock()
        mock_cascade.detectMultiScale.return_value = ()

        detector = EmotionDetector()
        detector._face_cascade = mock_cascade

        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        assert detector._score_frame(frame) == 0.0

    def test_score_frame_happy_face(self):
        """_score_frame should return 1.0 for happy face detection."""
        from clipshow.detection.emotion import EmotionDetector

        mock_cascade = MagicMock()
        mock_cascade.detectMultiScale.return_value = np.array([[10, 10, 50, 50]])

        mock_session = MagicMock()
        # happiness is index 1 — return high logit for it
        logits = np.array([[0, 10, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
        mock_session.run.return_value = [logits]
        mock_session.get_inputs.return_value = [MagicMock(name="Input3")]

        detector = EmotionDetector()
        detector._face_cascade = mock_cascade
        detector._emotion_session = mock_session

        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        assert detector._score_frame(frame) == 1.0

    def test_classify_emotion(self):
        """_classify_emotion should return correct label from logits."""
        from clipshow.detection.emotion import EmotionDetector

        mock_session = MagicMock()
        # surprise is index 2
        logits = np.array([[0, 0, 10, 0, 0, 0, 0, 0]], dtype=np.float32)
        mock_session.run.return_value = [logits]
        mock_session.get_inputs.return_value = [MagicMock(name="Input3")]

        detector = EmotionDetector()
        detector._emotion_session = mock_session

        gray_face = np.zeros((64, 64), dtype=np.uint8)
        assert detector._classify_emotion(gray_face) == "surprise"
