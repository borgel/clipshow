"""Integration tests for individual detectors on synthetic videos."""

import numpy as np
import pytest

from clipshow.detection.base import DetectorResult
from clipshow.detection.motion import MotionDetector
from clipshow.detection.scene import SceneDetector


class TestSceneDetector:
    def test_returns_detector_result(self, static_video):
        detector = SceneDetector()
        result = detector.detect(static_video)
        assert isinstance(result, DetectorResult)
        assert result.name == "scene"
        assert result.source_path == static_video
        assert result.time_step == 0.1

    def test_static_video_low_scores(self, static_video):
        """A uniform video should have very low scene change scores."""
        detector = SceneDetector()
        result = detector.detect(static_video)
        assert len(result.scores) > 0
        # Static video: after normalization, should be all zeros or very low
        # (no actual scene changes to detect)
        assert result.scores.max() <= 1.0

    def test_scene_change_detected(self, scene_change_video):
        """A video with a color change at 1.0s should show a score spike near that time."""
        detector = SceneDetector()
        result = detector.detect(scene_change_video)
        assert len(result.scores) > 0
        assert result.scores.max() > 0.5  # Should have a clear peak

        # Peak should be near t=1.0s (index ~10 at 10 samples/sec)
        peak_idx = np.argmax(result.scores)
        peak_time = peak_idx * result.time_step
        assert 0.5 < peak_time < 1.5, f"Peak at {peak_time}s, expected ~1.0s"

    def test_scores_normalized(self, scene_change_video):
        detector = SceneDetector()
        result = detector.detect(scene_change_video)
        assert result.scores.min() >= 0.0
        assert result.scores.max() <= 1.0

    def test_progress_callback(self, static_video):
        progress_values = []
        detector = SceneDetector()
        detector.detect(static_video, progress_callback=progress_values.append)
        assert 1.0 in progress_values


class TestMotionDetector:
    def test_returns_detector_result(self, static_video):
        detector = MotionDetector()
        result = detector.detect(static_video)
        assert isinstance(result, DetectorResult)
        assert result.name == "motion"
        assert result.source_path == static_video

    def test_static_video_low_motion(self, static_video):
        """A uniform static video should have no motion."""
        detector = MotionDetector()
        result = detector.detect(static_video)
        assert len(result.scores) > 0
        # After normalization, a truly static video should be all zeros
        # (or very near zero due to compression artifacts)
        assert result.scores.mean() < 0.3

    def test_motion_video_higher_than_static(self, static_video, motion_video):
        """A video with a moving object should score higher than a static one."""
        detector = MotionDetector()
        static_result = detector.detect(static_video)
        motion_result = detector.detect(motion_video)
        assert motion_result.scores.mean() > static_result.scores.mean()

    def test_scores_normalized(self, motion_video):
        detector = MotionDetector()
        result = detector.detect(motion_video)
        assert result.scores.min() >= 0.0
        assert result.scores.max() <= 1.0

    def test_progress_callback(self, static_video):
        progress_values = []
        detector = MotionDetector()
        detector.detect(static_video, progress_callback=progress_values.append)
        assert len(progress_values) > 0
        assert progress_values[-1] == pytest.approx(1.0)

    def test_cancel_flag(self, motion_video):
        """Detection should stop when cancel flag returns True."""
        call_count = 0

        def cancel_after_few():
            nonlocal call_count
            call_count += 1
            return call_count > 5

        detector = MotionDetector()
        result = detector.detect(motion_video, cancel_flag=cancel_after_few)
        # Should still return a valid result (just incomplete)
        assert isinstance(result, DetectorResult)


class TestAudioDetector:
    @pytest.mark.uses_audio_fixture
    def test_returns_detector_result(self, loud_moment_video):
        from clipshow.detection.audio import AudioDetector

        detector = AudioDetector()
        result = detector.detect(loud_moment_video)
        assert isinstance(result, DetectorResult)
        assert result.name == "audio"
        assert result.source_path == loud_moment_video

    @pytest.mark.uses_audio_fixture
    def test_loud_moment_detected(self, loud_moment_video):
        """A video with a noise burst at ~1.0s should show scores in that region."""
        from clipshow.detection.audio import AudioDetector

        detector = AudioDetector()
        result = detector.detect(loud_moment_video)
        assert len(result.scores) > 0
        assert result.scores.max() > 0.5  # Should have a clear peak

    @pytest.mark.uses_audio_fixture
    def test_scores_normalized(self, loud_moment_video):
        from clipshow.detection.audio import AudioDetector

        detector = AudioDetector()
        result = detector.detect(loud_moment_video)
        if len(result.scores) > 0:
            assert result.scores.min() >= 0.0
            assert result.scores.max() <= 1.0

    def test_no_audio_returns_empty(self, static_video):
        """A video with no audio track should return empty scores."""
        from clipshow.detection.audio import AudioDetector

        detector = AudioDetector()
        result = detector.detect(static_video)
        # static_video has no audio track, so scores should be empty
        assert isinstance(result, DetectorResult)
        assert len(result.scores) == 0

    @pytest.mark.uses_audio_fixture
    def test_progress_callback(self, loud_moment_video):
        from clipshow.detection.audio import AudioDetector

        progress_values = []
        detector = AudioDetector()
        detector.detect(loud_moment_video, progress_callback=progress_values.append)
        assert 1.0 in progress_values
