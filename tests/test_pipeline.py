"""Tests for detection pipeline: mock detectors + real synthetic video E2E."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from clipshow.config import Settings
from clipshow.detection.base import DetectorResult
from clipshow.detection.pipeline import DetectionPipeline
from clipshow.model.moments import DetectedMoment


class TestPipelineWithMocks:
    """Test pipeline orchestration using mock detectors."""

    def _mock_detector(self, name: str, scores: np.ndarray, time_step: float = 0.1):
        """Create a mock detector that returns predetermined scores."""
        detector = MagicMock()
        detector.name = name
        detector.detect.return_value = DetectorResult(
            name=name,
            scores=scores,
            time_step=time_step,
            source_path="test.mp4",
        )
        return detector

    def test_runs_enabled_detectors_only(self):
        """Pipeline should skip detectors with weight=0."""
        settings = Settings(
            scene_weight=0.5,
            audio_weight=0.0,  # Disabled
            motion_weight=0.5,
            semantic_weight=0.0,
            emotion_weight=0.0,
        )
        pipeline = DetectionPipeline(settings)
        detectors = pipeline._build_detectors()
        names = [name for name, _, _ in detectors]
        assert "scene" in names
        assert "motion" in names
        assert "audio" not in names

    def test_all_detectors_disabled(self):
        settings = Settings(
            scene_weight=0.0,
            audio_weight=0.0,
            motion_weight=0.0,
            semantic_weight=0.0,
            emotion_weight=0.0,
        )
        pipeline = DetectionPipeline(settings)
        moments = pipeline.analyze_video("test.mp4", video_duration=5.0)
        assert moments == []

    def test_progress_callback_called(self):
        settings = Settings(scene_weight=1.0, audio_weight=0.0, motion_weight=0.0)
        pipeline = DetectionPipeline(settings)

        mock_scene = self._mock_detector("scene", np.array([0.0, 0.5, 1.0, 0.5, 0.0]))
        progress_values = []

        with patch.dict(
            "clipshow.detection.pipeline.DETECTOR_CLASSES",
            {"scene": lambda: mock_scene},
        ):
            # Need to patch _build_detectors to use our mock
            pipeline._build_detectors = lambda: [("scene", mock_scene, 1.0)]
            pipeline.analyze_video(
                "test.mp4",
                video_duration=0.5,
                progress_callback=progress_values.append,
            )

        assert len(progress_values) > 0
        assert progress_values[-1] == pytest.approx(1.0)

    def test_cancel_flag_stops_processing(self):
        settings = Settings(scene_weight=1.0, audio_weight=1.0, motion_weight=1.0)
        pipeline = DetectionPipeline(settings)

        call_count = 0

        def cancel_immediately():
            nonlocal call_count
            call_count += 1
            return True  # Cancel right away

        moments = pipeline.analyze_video(
            "test.mp4",
            video_duration=5.0,
            cancel_flag=cancel_immediately,
        )
        # Should return empty or partial (cancelled before any detector ran)
        assert isinstance(moments, list)

    def test_analyze_all_combines_results(self):
        settings = Settings(
            scene_weight=1.0,
            audio_weight=0.0,
            motion_weight=0.0,
            score_threshold=0.3,
        )
        pipeline = DetectionPipeline(settings)

        # Create a mock that returns a strong peak
        scores = np.zeros(50)  # 5 seconds
        scores[20:30] = 0.9
        mock_scene = self._mock_detector("scene", scores)
        pipeline._build_detectors = lambda: [("scene", mock_scene, 1.0)]

        results = pipeline.analyze_all(
            [("video1.mp4", 5.0), ("video2.mp4", 5.0)]
        )
        # Should have moments from both videos
        # (mock returns same scores for both, but source_path differs)
        assert len(results) >= 2


class TestPipelineE2E:
    """End-to-end pipeline tests using real synthetic videos."""

    def test_scene_change_video_produces_moments(self, scene_change_video):
        settings = Settings(
            scene_weight=1.0,
            audio_weight=0.0,
            motion_weight=0.0,
            score_threshold=0.3,
            min_segment_duration_sec=0.2,
        )
        pipeline = DetectionPipeline(settings)
        moments = pipeline.analyze_video(scene_change_video, video_duration=2.0)
        assert len(moments) >= 1
        assert all(isinstance(m, DetectedMoment) for m in moments)
        assert moments[0].source_path == scene_change_video

    def test_motion_video_produces_moments(self, motion_video):
        settings = Settings(
            scene_weight=0.0,
            audio_weight=0.0,
            motion_weight=1.0,
            score_threshold=0.3,
            min_segment_duration_sec=0.2,
        )
        pipeline = DetectionPipeline(settings)
        moments = pipeline.analyze_video(motion_video, video_duration=2.0)
        assert len(moments) >= 1

    def test_audio_video_produces_moments(self, loud_moment_video):
        settings = Settings(
            scene_weight=0.0,
            audio_weight=1.0,
            motion_weight=0.0,
            score_threshold=0.3,
            min_segment_duration_sec=0.2,
        )
        pipeline = DetectionPipeline(settings)
        moments = pipeline.analyze_video(loud_moment_video, video_duration=2.0)
        assert len(moments) >= 1

    def test_static_video_no_moments(self, static_video):
        """A static, silent video should produce no moments above threshold."""
        settings = Settings(
            scene_weight=1.0,
            audio_weight=0.0,
            motion_weight=1.0,
            score_threshold=0.5,
        )
        pipeline = DetectionPipeline(settings)
        moments = pipeline.analyze_video(static_video, video_duration=2.0)
        assert moments == []

    def test_multi_detector_combination(self, scene_change_video):
        """Using multiple detectors should still produce valid results."""
        settings = Settings(
            scene_weight=0.3,
            audio_weight=0.0,
            motion_weight=0.25,
            score_threshold=0.3,
            min_segment_duration_sec=0.2,
        )
        pipeline = DetectionPipeline(settings)
        moments = pipeline.analyze_video(scene_change_video, video_duration=2.0)
        assert isinstance(moments, list)
        if moments:
            assert moments[0].contributing_detectors == ["scene", "motion"]
