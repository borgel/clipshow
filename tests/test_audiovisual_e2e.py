"""End-to-end pipeline tests with AudioVisualDetector integration.

TDD tests written BEFORE pipeline integration (red phase). Verify the
AudioVisualDetector works within the full DetectionPipeline alongside
existing detectors. All ONNX sessions are mocked — no real model downloads.
Tests will initially fail (red); clipshow-ebo implements to make them pass.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from clipshow.config import Settings
from clipshow.detection.base import DetectorResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 768


def _mock_audiovisual_detect(video_path, progress_callback=None, cancel_flag=None):
    """Fake detect() that returns realistic-looking DetectorResult."""
    # Simulate 2s of scores at 0.1s timestep = 20 samples
    scores = np.zeros(20, dtype=float)
    # Spike at the middle (simulating an interesting moment)
    scores[8:12] = np.array([0.6, 0.9, 0.8, 0.5])
    if progress_callback:
        progress_callback(1.0)
    return DetectorResult(
        name="audiovisual",
        scores=scores,
        time_step=0.1,
        source_path=video_path,
    )


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------


class TestConfigAudiovisual:
    """Settings should support audiovisual detector configuration."""

    def test_audiovisual_weight_exists(self):
        """Settings should have an audiovisual_weight field."""
        s = Settings()
        assert hasattr(s, "audiovisual_weight")

    def test_audiovisual_weight_default_zero(self):
        """audiovisual_weight should default to 0 (disabled)."""
        s = Settings()
        assert s.audiovisual_weight == 0.0

    def test_audiovisual_in_detector_weights(self):
        """detector_weights property should include audiovisual."""
        s = Settings(audiovisual_weight=0.5)
        assert "audiovisual" in s.detector_weights
        assert s.detector_weights["audiovisual"] == 0.5

    def test_audiovisual_in_enabled_detectors(self):
        """enabled_detectors should list audiovisual when weight > 0."""
        s = Settings(audiovisual_weight=0.5)
        assert "audiovisual" in s.enabled_detectors

    def test_audiovisual_not_enabled_when_zero(self):
        """audiovisual should not appear in enabled_detectors when weight = 0."""
        s = Settings()
        assert "audiovisual" not in s.enabled_detectors

    def test_config_round_trip(self, tmp_path):
        """audiovisual_weight should survive save/load cycle."""
        config_path = tmp_path / "settings.json"
        s = Settings(audiovisual_weight=0.7)
        s.save(config_path)
        loaded = Settings.load(config_path)
        assert loaded.audiovisual_weight == 0.7


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


class TestPipelineWithAudiovisual:
    """DetectionPipeline should support audiovisual detector."""

    def test_audiovisual_detector_invoked_when_enabled(self, static_video):
        """Pipeline should invoke audiovisual detector when weight > 0."""
        from clipshow.detection.pipeline import DetectionPipeline

        settings = Settings(audiovisual_weight=1.0, score_threshold=0.3)

        mock_detector_cls = MagicMock()
        mock_instance = MagicMock()
        mock_instance.detect.side_effect = _mock_audiovisual_detect
        mock_detector_cls.return_value = mock_instance

        pipeline = DetectionPipeline(settings)

        with patch(
            "clipshow.detection.pipeline._get_optional_detector"
        ) as mock_get:
            # Return our mock class when "audiovisual" is requested
            def get_detector(name):
                if name == "audiovisual":
                    return mock_detector_cls
                return None

            mock_get.side_effect = get_detector
            pipeline.analyze_video(static_video, video_duration=2.0)

        mock_instance.detect.assert_called_once()

    def test_audiovisual_skipped_when_weight_zero(self, static_video):
        """Pipeline should skip audiovisual when weight = 0."""
        from clipshow.detection.pipeline import DetectionPipeline

        settings = Settings(audiovisual_weight=0.0)
        pipeline = DetectionPipeline(settings)

        mock_detector_cls = MagicMock()

        with patch(
            "clipshow.detection.pipeline._get_optional_detector"
        ) as mock_get:
            mock_get.return_value = mock_detector_cls
            pipeline.analyze_video(static_video, video_duration=2.0)

        mock_detector_cls.assert_not_called()

    def test_audiovisual_skipped_when_import_fails(self, static_video):
        """Pipeline should skip audiovisual gracefully if deps missing."""
        from clipshow.detection.pipeline import DetectionPipeline

        settings = Settings(audiovisual_weight=1.0)
        pipeline = DetectionPipeline(settings)
        warnings = []

        with patch(
            "clipshow.detection.pipeline._get_optional_detector",
            return_value=None,
        ):
            moments = pipeline.analyze_video(
                static_video,
                video_duration=2.0,
                warning_callback=warnings.append,
            )

        # Should produce empty results (no working detectors)
        assert isinstance(moments, list)

    def test_audiovisual_combined_with_other_detectors(self, static_video):
        """audiovisual scores should be combined with other detectors."""
        from clipshow.detection.pipeline import DetectionPipeline

        settings = Settings(
            audiovisual_weight=0.5,
            motion_weight=0.5,
            score_threshold=0.3,
        )

        # Mock both detectors
        mock_av_cls = MagicMock()
        mock_av_instance = MagicMock()
        mock_av_instance.detect.side_effect = _mock_audiovisual_detect
        mock_av_cls.return_value = mock_av_instance

        # Mock motion detector
        def mock_motion_detect(
            video_path, progress_callback=None, cancel_flag=None
        ):
            scores = np.zeros(20, dtype=float)
            scores[5:10] = 0.7  # Different pattern than audiovisual
            if progress_callback:
                progress_callback(1.0)
            return DetectorResult(
                name="motion",
                scores=scores,
                time_step=0.1,
                source_path=video_path,
            )

        mock_motion_cls = MagicMock()
        mock_motion_instance = MagicMock()
        mock_motion_instance.detect.side_effect = mock_motion_detect
        mock_motion_cls.return_value = mock_motion_instance

        pipeline = DetectionPipeline(settings)

        with patch.dict(
            "clipshow.detection.pipeline.DETECTOR_CLASSES",
            {"motion": mock_motion_cls},
            clear=True,
        ):
            with patch(
                "clipshow.detection.pipeline._get_optional_detector"
            ) as mock_get:
                mock_get.side_effect = lambda n: (
                    mock_av_cls if n == "audiovisual" else None
                )
                pipeline.analyze_video(
                    static_video, video_duration=2.0
                )

        # Both detectors should have been called
        mock_av_instance.detect.assert_called_once()
        mock_motion_instance.detect.assert_called_once()

    def test_contributing_detectors_includes_audiovisual(self, static_video):
        """DetectedMoments should list audiovisual in contributing_detectors."""
        from clipshow.detection.pipeline import DetectionPipeline

        settings = Settings(audiovisual_weight=1.0, score_threshold=0.3)

        mock_av_cls = MagicMock()
        mock_av_instance = MagicMock()
        mock_av_instance.detect.side_effect = _mock_audiovisual_detect
        mock_av_cls.return_value = mock_av_instance

        pipeline = DetectionPipeline(settings)

        with patch.dict(
            "clipshow.detection.pipeline.DETECTOR_CLASSES", {}, clear=True
        ):
            with patch(
                "clipshow.detection.pipeline._get_optional_detector"
            ) as mock_get:
                mock_get.side_effect = lambda n: (
                    mock_av_cls if n == "audiovisual" else None
                )
                moments = pipeline.analyze_video(
                    static_video, video_duration=2.0
                )

        assert len(moments) > 0
        for m in moments:
            assert "audiovisual" in m.contributing_detectors

    def test_audiovisual_scores_in_valid_range(self, static_video):
        """All detected moments should have scores in [0, 1]."""
        from clipshow.detection.pipeline import DetectionPipeline

        settings = Settings(audiovisual_weight=1.0, score_threshold=0.3)

        mock_av_cls = MagicMock()
        mock_av_instance = MagicMock()
        mock_av_instance.detect.side_effect = _mock_audiovisual_detect
        mock_av_cls.return_value = mock_av_instance

        pipeline = DetectionPipeline(settings)

        with patch.dict(
            "clipshow.detection.pipeline.DETECTOR_CLASSES", {}, clear=True
        ):
            with patch(
                "clipshow.detection.pipeline._get_optional_detector"
            ) as mock_get:
                mock_get.side_effect = lambda n: (
                    mock_av_cls if n == "audiovisual" else None
                )
                moments = pipeline.analyze_video(
                    static_video, video_duration=2.0
                )

        for m in moments:
            assert 0.0 <= m.peak_score <= 1.0
            assert 0.0 <= m.mean_score <= 1.0


# ---------------------------------------------------------------------------
# Regression — existing detectors still work
# ---------------------------------------------------------------------------


class TestNoRegressions:
    """Verify adding audiovisual doesn't break existing detector flow."""

    def test_pipeline_works_without_audiovisual(self, static_video):
        """Pipeline should still work with only traditional detectors."""
        from clipshow.detection.pipeline import DetectionPipeline

        # Only motion enabled, no audiovisual
        settings = Settings(motion_weight=1.0, score_threshold=0.3)

        def mock_motion_detect(
            video_path, progress_callback=None, cancel_flag=None
        ):
            scores = np.ones(20, dtype=float) * 0.8
            if progress_callback:
                progress_callback(1.0)
            return DetectorResult(
                name="motion",
                scores=scores,
                time_step=0.1,
                source_path=video_path,
            )

        mock_cls = MagicMock()
        mock_instance = MagicMock()
        mock_instance.detect.side_effect = mock_motion_detect
        mock_cls.return_value = mock_instance

        pipeline = DetectionPipeline(settings)

        with patch.dict(
            "clipshow.detection.pipeline.DETECTOR_CLASSES",
            {"motion": mock_cls},
            clear=True,
        ):
            moments = pipeline.analyze_video(static_video, video_duration=2.0)

        assert len(moments) > 0

    def test_empty_pipeline_returns_empty(self):
        """Pipeline with no enabled detectors should return empty list."""
        from clipshow.detection.pipeline import DetectionPipeline

        settings = Settings()  # All weights = 0
        pipeline = DetectionPipeline(settings)
        moments = pipeline.analyze_video("/fake/video.mp4", video_duration=2.0)
        assert moments == []
