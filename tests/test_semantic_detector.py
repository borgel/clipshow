"""Tests for semantic detector (CLIP-based) with mocked onnx_clip."""

import importlib
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from clipshow.detection.base import DetectorResult


class TestSemanticDetector:
    def test_missing_onnx_clip_raises(self):
        """Should raise RuntimeError if onnx_clip is not installed."""
        from clipshow.detection.semantic import SemanticDetector

        original_import = importlib.import_module

        def selective_import(name):
            if name == "onnx_clip":
                raise ImportError("no onnx_clip")
            return original_import(name)

        detector = SemanticDetector()
        with patch("importlib.import_module", side_effect=selective_import):
            with pytest.raises(RuntimeError, match="onnx_clip is not installed"):
                detector._load_model()

    def test_detect_returns_detector_result(self, static_video):
        """With mocked model, detect should return a valid DetectorResult."""
        from clipshow.detection.semantic import SemanticDetector

        mock_model = MagicMock()
        mock_model.get_text_embeddings.return_value = np.ones((4, 512))
        mock_model.get_image_embeddings.return_value = np.ones((1, 512)) * 0.5

        detector = SemanticDetector()
        detector._model = mock_model

        mock_image_mod = MagicMock()
        mock_image_mod.fromarray.return_value = MagicMock()
        pil_mods = {
            "PIL.Image": mock_image_mod,
            "PIL": MagicMock(Image=mock_image_mod),
        }
        with patch.dict("sys.modules", pil_mods):
            result = detector.detect(static_video)

        assert isinstance(result, DetectorResult)
        assert result.name == "semantic"
        assert result.source_path == static_video
        assert len(result.scores) > 0
        assert result.scores.min() >= 0.0
        assert result.scores.max() <= 1.0

    def test_cancel_flag_stops_processing(self, static_video):
        """Cancel flag should stop frame processing."""
        from clipshow.detection.semantic import SemanticDetector

        mock_model = MagicMock()
        mock_model.get_text_embeddings.return_value = np.ones((4, 512))
        mock_model.get_image_embeddings.return_value = np.ones((1, 512)) * 0.5

        detector = SemanticDetector()
        detector._model = mock_model

        mock_image_mod = MagicMock()
        mock_image_mod.fromarray.return_value = MagicMock()
        # Cancel immediately
        pil_mods = {
            "PIL.Image": mock_image_mod,
            "PIL": MagicMock(Image=mock_image_mod),
        }
        with patch.dict("sys.modules", pil_mods):
            result = detector.detect(static_video, cancel_flag=lambda: True)

        assert isinstance(result, DetectorResult)

    def test_default_prompts(self):
        from clipshow.detection.semantic import DEFAULT_PROMPTS, SemanticDetector

        detector = SemanticDetector()
        assert detector._prompts == DEFAULT_PROMPTS

    def test_custom_prompts(self):
        from clipshow.detection.semantic import SemanticDetector

        prompts = ["sports action", "celebration"]
        detector = SemanticDetector(prompts=prompts)
        assert detector._prompts == prompts


class TestPipelineLazyLoading:
    def test_semantic_detector_loaded_by_pipeline(self):
        """Pipeline should find SemanticDetector via lazy loading."""
        from clipshow.detection.pipeline import _get_optional_detector

        cls = _get_optional_detector("semantic")
        # Should return the class (onnx_clip may not be installed,
        # but the module itself should import fine)
        assert cls is not None
        assert cls.name == "semantic"

    def test_emotion_detector_loaded_by_pipeline(self):
        """Pipeline should find EmotionDetector via lazy loading."""
        from clipshow.detection.pipeline import _get_optional_detector

        cls = _get_optional_detector("emotion")
        assert cls is not None
        assert cls.name == "emotion"

    def test_unknown_detector_returns_none(self):
        from clipshow.detection.pipeline import _get_optional_detector

        assert _get_optional_detector("unknown") is None
