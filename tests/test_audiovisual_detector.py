"""TDD tests for AudioVisualDetector (clipshow/detection/audiovisual.py).

Tests written BEFORE implementation (red phase). Define the expected
interface and behavior for the LanguageBind-based audio-visual-text
detector. All ONNX sessions are mocked — no real model downloads in CI.
Tests will initially fail (red); clipshow-7lq implements to make them pass.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from clipshow.detection.audiovisual import (
    DEFAULT_NEGATIVE_PROMPTS,
    DEFAULT_PROMPTS,
    AudioVisualDetector,
)
from clipshow.detection.base import DetectorResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 768  # LanguageBind embedding dimension


def _mock_onnx_session(output_shape):
    """Create a mock InferenceSession that returns random embeddings."""
    session = MagicMock()
    rng = np.random.default_rng(42)

    def run_side_effect(output_names, input_dict):
        # Determine batch size from first input
        first_input = next(iter(input_dict.values()))
        batch = first_input.shape[0]
        emb = rng.standard_normal((batch, output_shape)).astype(np.float32)
        # L2 normalize (matching LanguageBind output)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.maximum(norms, 1e-8)
        return [emb]

    session.run.side_effect = run_side_effect
    return session


def _make_detector_with_mock_sessions(**kwargs):
    """Create an AudioVisualDetector with all three sessions pre-mocked."""
    detector = AudioVisualDetector(**kwargs)
    detector._video_session = _mock_onnx_session(EMBEDDING_DIM)
    detector._audio_session = _mock_onnx_session(EMBEDDING_DIM)
    detector._text_session = _mock_onnx_session(EMBEDDING_DIM)
    return detector


# ---------------------------------------------------------------------------
# Construction and defaults
# ---------------------------------------------------------------------------


class TestAudioVisualDetectorInit:
    def test_name(self):
        detector = AudioVisualDetector()
        assert detector.name == "audiovisual"

    def test_default_prompts(self):
        detector = AudioVisualDetector()
        assert detector._prompts == DEFAULT_PROMPTS

    def test_default_negative_prompts(self):
        detector = AudioVisualDetector()
        assert detector._negative_prompts == DEFAULT_NEGATIVE_PROMPTS

    def test_custom_prompts(self):
        prompts = ["sports action", "celebration"]
        detector = AudioVisualDetector(prompts=prompts)
        assert detector._prompts == prompts

    def test_custom_negative_prompts(self):
        neg = ["static shot"]
        detector = AudioVisualDetector(negative_prompts=neg)
        assert detector._negative_prompts == neg

    def test_default_audio_weight(self):
        detector = AudioVisualDetector()
        assert detector._audio_weight == 0.4

    def test_custom_audio_weight(self):
        detector = AudioVisualDetector(audio_weight=0.6)
        assert detector._audio_weight == 0.6

    def test_sessions_none_before_load(self):
        detector = AudioVisualDetector()
        assert detector._video_session is None
        assert detector._audio_session is None
        assert detector._text_session is None


# ---------------------------------------------------------------------------
# _load_models
# ---------------------------------------------------------------------------


class TestLoadModels:
    def test_loads_three_sessions(self):
        """_load_models should populate all three ONNX sessions."""
        detector = AudioVisualDetector()

        mock_manager = MagicMock()
        mock_manager.load_session.return_value = MagicMock()

        with patch(
            "clipshow.detection.audiovisual.ModelManager",
            return_value=mock_manager,
        ):
            detector._load_models()

        assert detector._video_session is not None
        assert detector._audio_session is not None
        assert detector._text_session is not None

    def test_loads_correct_model_names(self):
        """Should load languagebind-video, languagebind-audio, languagebind-text."""
        detector = AudioVisualDetector()

        mock_manager = MagicMock()
        mock_manager.load_session.return_value = MagicMock()

        with patch(
            "clipshow.detection.audiovisual.ModelManager",
            return_value=mock_manager,
        ):
            detector._load_models()

        loaded_names = [
            call.args[0] for call in mock_manager.load_session.call_args_list
        ]
        assert "languagebind-video" in loaded_names
        assert "languagebind-audio" in loaded_names
        assert "languagebind-text" in loaded_names


# ---------------------------------------------------------------------------
# detect() — full pipeline with mocked sessions
# ---------------------------------------------------------------------------


class TestDetect:
    """Test detect() with mocked ONNX sessions on synthetic video."""

    def test_returns_detector_result(self, static_video):
        """detect() should return a DetectorResult."""
        detector = _make_detector_with_mock_sessions()

        with patch(
            "clipshow.detection.audiovisual.extract_mel_windows",
            return_value=np.random.randn(5, 128, 101).astype(np.float32),
        ):
            with patch(
                "clipshow.detection.audiovisual.format_for_languagebind",
                return_value=np.random.randn(5, 3, 128, 1036).astype(
                    np.float32
                ),
            ):
                result = detector.detect(static_video)

        assert isinstance(result, DetectorResult)

    def test_result_name(self, static_video):
        """Result name should be 'audiovisual'."""
        detector = _make_detector_with_mock_sessions()

        with patch(
            "clipshow.detection.audiovisual.extract_mel_windows",
            return_value=np.random.randn(5, 128, 101).astype(np.float32),
        ):
            with patch(
                "clipshow.detection.audiovisual.format_for_languagebind",
                return_value=np.random.randn(5, 3, 128, 1036).astype(
                    np.float32
                ),
            ):
                result = detector.detect(static_video)

        assert result.name == "audiovisual"

    def test_result_source_path(self, static_video):
        """Result source_path should match input video."""
        detector = _make_detector_with_mock_sessions()

        with patch(
            "clipshow.detection.audiovisual.extract_mel_windows",
            return_value=np.random.randn(5, 128, 101).astype(np.float32),
        ):
            with patch(
                "clipshow.detection.audiovisual.format_for_languagebind",
                return_value=np.random.randn(5, 3, 128, 1036).astype(
                    np.float32
                ),
            ):
                result = detector.detect(static_video)

        assert result.source_path == static_video

    def test_scores_in_zero_one_range(self, static_video):
        """All scores should be in [0, 1]."""
        detector = _make_detector_with_mock_sessions()

        with patch(
            "clipshow.detection.audiovisual.extract_mel_windows",
            return_value=np.random.randn(5, 128, 101).astype(np.float32),
        ):
            with patch(
                "clipshow.detection.audiovisual.format_for_languagebind",
                return_value=np.random.randn(5, 3, 128, 1036).astype(
                    np.float32
                ),
            ):
                result = detector.detect(static_video)

        assert len(result.scores) > 0
        assert result.scores.min() >= 0.0
        assert result.scores.max() <= 1.0

    def test_scores_length_matches_duration(self, static_video):
        """Number of scores should match video duration / time_step."""
        detector = _make_detector_with_mock_sessions(time_step=0.1)

        with patch(
            "clipshow.detection.audiovisual.extract_mel_windows",
            return_value=np.random.randn(5, 128, 101).astype(np.float32),
        ):
            with patch(
                "clipshow.detection.audiovisual.format_for_languagebind",
                return_value=np.random.randn(5, 3, 128, 1036).astype(
                    np.float32
                ),
            ):
                result = detector.detect(static_video)

        # Static video is 2s, time_step=0.1 -> ~20 samples
        assert result.num_samples >= 15  # allow some tolerance
        assert result.num_samples <= 25

    def test_time_step_matches(self, static_video):
        """Result time_step should match detector's configured value."""
        detector = _make_detector_with_mock_sessions(time_step=0.2)

        with patch(
            "clipshow.detection.audiovisual.extract_mel_windows",
            return_value=np.random.randn(5, 128, 101).astype(np.float32),
        ):
            with patch(
                "clipshow.detection.audiovisual.format_for_languagebind",
                return_value=np.random.randn(5, 3, 128, 1036).astype(
                    np.float32
                ),
            ):
                result = detector.detect(static_video)

        assert result.time_step == 0.2

    def test_handles_video_without_audio(self, static_video):
        """Should still work when video has no audio track."""
        detector = _make_detector_with_mock_sessions()

        # extract_mel_windows returns None for no-audio videos
        with patch(
            "clipshow.detection.audiovisual.extract_mel_windows",
            return_value=None,
        ):
            result = detector.detect(static_video)

        assert isinstance(result, DetectorResult)
        assert result.scores.min() >= 0.0
        assert result.scores.max() <= 1.0

    def test_lazy_loads_models(self, static_video):
        """First call to detect() should trigger _load_models if sessions are None."""
        detector = AudioVisualDetector()

        mock_manager = MagicMock()
        mock_session = _mock_onnx_session(EMBEDDING_DIM)
        mock_manager.load_session.return_value = mock_session

        with patch(
            "clipshow.detection.audiovisual.ModelManager",
            return_value=mock_manager,
        ):
            with patch(
                "clipshow.detection.audiovisual.extract_mel_windows",
                return_value=np.random.randn(5, 128, 101).astype(np.float32),
            ):
                with patch(
                    "clipshow.detection.audiovisual.format_for_languagebind",
                    return_value=np.random.randn(5, 3, 128, 1036).astype(
                        np.float32
                    ),
                ):
                    result = detector.detect(static_video)

        assert isinstance(result, DetectorResult)
        assert mock_manager.load_session.call_count == 3


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


class TestCancellation:
    def test_cancel_flag_stops_processing(self, static_video):
        """Cancel flag should stop processing and return partial result."""
        detector = _make_detector_with_mock_sessions()

        with patch(
            "clipshow.detection.audiovisual.extract_mel_windows",
            return_value=np.random.randn(5, 128, 101).astype(np.float32),
        ):
            with patch(
                "clipshow.detection.audiovisual.format_for_languagebind",
                return_value=np.random.randn(5, 3, 128, 1036).astype(
                    np.float32
                ),
            ):
                result = detector.detect(
                    static_video, cancel_flag=lambda: True
                )

        assert isinstance(result, DetectorResult)


# ---------------------------------------------------------------------------
# Fusion math
# ---------------------------------------------------------------------------


class TestFusionMath:
    """Test the audio-video similarity fusion logic."""

    def test_audio_weight_affects_fusion(self, static_video):
        """Different audio_weight values should produce different scores."""
        dim = 16
        rng = np.random.default_rng(42)

        # Create structured embeddings where video and audio produce
        # different similarity patterns with text prompts.
        video_emb = rng.standard_normal((5, dim)).astype(np.float32)
        video_emb /= np.linalg.norm(video_emb, axis=1, keepdims=True)
        audio_emb = rng.standard_normal((5, dim)).astype(np.float32)
        audio_emb /= np.linalg.norm(audio_emb, axis=1, keepdims=True)

        # CRITICAL: positive and negative text embeddings must DIFFER,
        # otherwise raw = max_pos - max_neg = 0 regardless of fusion.
        text_pos = rng.standard_normal((4, dim)).astype(np.float32)
        text_pos /= np.linalg.norm(text_pos, axis=1, keepdims=True)
        text_neg = rng.standard_normal((4, dim)).astype(np.float32)
        text_neg /= np.linalg.norm(text_neg, axis=1, keepdims=True)

        def run_with_weight(weight):
            detector = _make_detector_with_mock_sessions(audio_weight=weight)
            detector._video_session.run.side_effect = None
            detector._audio_session.run.side_effect = None

            detector._video_session.run.return_value = [video_emb.copy()]
            detector._audio_session.run.return_value = [audio_emb.copy()]
            # Text session is called twice: once for pos, once for neg prompts
            detector._text_session.run.side_effect = [
                [text_pos.copy()],
                [text_neg.copy()],
            ]

            with patch(
                "clipshow.detection.audiovisual.extract_mel_windows",
                return_value=np.random.randn(5, 128, 101).astype(np.float32),
            ):
                with patch(
                    "clipshow.detection.audiovisual.format_for_languagebind",
                    return_value=np.random.randn(5, 3, 128, 1036).astype(
                        np.float32
                    ),
                ):
                    return detector.detect(static_video)

        result_low = run_with_weight(0.1)
        result_high = run_with_weight(0.9)

        # Scores should differ when audio weight changes
        assert not np.allclose(
            result_low.scores, result_high.scores
        ), "Different audio weights should produce different scores"

    def test_audio_weight_zero_ignores_audio(self, static_video):
        """With audio_weight=0.0, scores should depend only on video."""
        detector = _make_detector_with_mock_sessions(audio_weight=0.0)

        with patch(
            "clipshow.detection.audiovisual.extract_mel_windows",
            return_value=np.random.randn(5, 128, 101).astype(np.float32),
        ):
            with patch(
                "clipshow.detection.audiovisual.format_for_languagebind",
                return_value=np.random.randn(5, 3, 128, 1036).astype(
                    np.float32
                ),
            ):
                result = detector.detect(static_video)

        # Should still produce valid scores
        assert isinstance(result, DetectorResult)
        assert result.scores.min() >= 0.0

    def test_audio_weight_one_ignores_video(self, static_video):
        """With audio_weight=1.0, scores should depend only on audio."""
        detector = _make_detector_with_mock_sessions(audio_weight=1.0)

        with patch(
            "clipshow.detection.audiovisual.extract_mel_windows",
            return_value=np.random.randn(5, 128, 101).astype(np.float32),
        ):
            with patch(
                "clipshow.detection.audiovisual.format_for_languagebind",
                return_value=np.random.randn(5, 3, 128, 1036).astype(
                    np.float32
                ),
            ):
                result = detector.detect(static_video)

        assert isinstance(result, DetectorResult)
        assert result.scores.min() >= -1e-10  # allow tiny floating-point rounding


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------


class TestProgressCallback:
    def test_progress_callback_called(self, static_video):
        """Progress callback should be invoked during detect()."""
        detector = _make_detector_with_mock_sessions()
        progress_values = []

        with patch(
            "clipshow.detection.audiovisual.extract_mel_windows",
            return_value=np.random.randn(5, 128, 101).astype(np.float32),
        ):
            with patch(
                "clipshow.detection.audiovisual.format_for_languagebind",
                return_value=np.random.randn(5, 3, 128, 1036).astype(
                    np.float32
                ),
            ):
                detector.detect(
                    static_video,
                    progress_callback=lambda p: progress_values.append(p),
                )

        assert len(progress_values) > 0

    def test_progress_ends_at_one(self, static_video):
        """Final progress value should be 1.0."""
        detector = _make_detector_with_mock_sessions()
        progress_values = []

        with patch(
            "clipshow.detection.audiovisual.extract_mel_windows",
            return_value=np.random.randn(5, 128, 101).astype(np.float32),
        ):
            with patch(
                "clipshow.detection.audiovisual.format_for_languagebind",
                return_value=np.random.randn(5, 3, 128, 1036).astype(
                    np.float32
                ),
            ):
                detector.detect(
                    static_video,
                    progress_callback=lambda p: progress_values.append(p),
                )

        assert progress_values[-1] == pytest.approx(1.0)
