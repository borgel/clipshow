"""Tests for detection/scoring.py — pure numpy, no video files needed."""

import numpy as np
import pytest

from clipshow.detection.scoring import (
    COMMON_TIME_STEP,
    apply_padding,
    extract_moments,
    filter_by_duration,
    merge_overlapping,
    resample_scores,
    threshold_segments,
    weighted_combine,
)


class TestResampleScores:
    def test_identity_when_same_rate(self):
        """Resampling at the common rate should preserve values."""
        scores = np.array([0.0, 0.5, 1.0, 0.5, 0.0])
        result = resample_scores(scores, COMMON_TIME_STEP)
        np.testing.assert_allclose(result, scores, atol=0.01)

    def test_upsample(self):
        """Resampling from 1 sample/sec to 10 samples/sec."""
        scores = np.array([0.0, 1.0])  # 2 samples at 1Hz = 2 seconds
        result = resample_scores(scores, source_time_step=1.0)
        assert len(result) == 20  # 2s * 10 samples/sec
        assert result[0] == pytest.approx(0.0)
        assert result[-1] == pytest.approx(1.0)
        # Should be linearly interpolated
        assert result[10] == pytest.approx(1.0, abs=0.05)

    def test_downsample(self):
        """Resampling from 100 samples/sec to 10 samples/sec."""
        scores = np.linspace(0, 1, 100)  # 100 samples at 100Hz = 1 second
        result = resample_scores(scores, source_time_step=0.01)
        assert len(result) == 10  # 1s * 10 samples/sec

    def test_empty_input(self):
        result = resample_scores(np.array([]), 0.1)
        assert len(result) == 0


class TestWeightedCombine:
    def test_equal_weights(self):
        a = np.array([1.0, 0.0, 0.5])
        b = np.array([0.0, 1.0, 0.5])
        result = weighted_combine([a, b], [1.0, 1.0])
        # (1+0)/2=0.5, (0+1)/2=0.5, (0.5+0.5)/2=0.5 → all same → renorm to 1.0
        np.testing.assert_allclose(result, [1.0, 1.0, 1.0])

    def test_single_detector(self):
        a = np.array([0.0, 0.5, 1.0])
        result = weighted_combine([a], [1.0])
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])

    def test_unequal_weights(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        result = weighted_combine([a, b], [3.0, 1.0])
        # weighted: [0.75, 0.25] → renorm: [1.0, 0.333]
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(1.0 / 3.0, abs=0.01)

    def test_zero_weights(self):
        a = np.array([1.0, 0.5])
        result = weighted_combine([a], [0.0])
        np.testing.assert_allclose(result, [0.0, 0.0])

    def test_empty_input(self):
        result = weighted_combine([], [])
        assert len(result) == 0


class TestThresholdSegments:
    def test_single_peak(self):
        scores = np.array([0.0, 0.0, 0.8, 0.9, 0.7, 0.0, 0.0])
        segments = threshold_segments(scores, 0.5)
        assert segments == [(2, 5)]

    def test_two_peaks(self):
        scores = np.array([0.0, 0.8, 0.0, 0.0, 0.9, 0.0])
        segments = threshold_segments(scores, 0.5)
        assert segments == [(1, 2), (4, 5)]

    def test_starts_above(self):
        scores = np.array([0.8, 0.9, 0.0, 0.0])
        segments = threshold_segments(scores, 0.5)
        assert segments == [(0, 2)]

    def test_ends_above(self):
        scores = np.array([0.0, 0.0, 0.8, 0.9])
        segments = threshold_segments(scores, 0.5)
        assert segments == [(2, 4)]

    def test_all_above(self):
        scores = np.array([0.8, 0.9, 0.7, 0.6])
        segments = threshold_segments(scores, 0.5)
        assert segments == [(0, 4)]

    def test_none_above(self):
        scores = np.array([0.1, 0.2, 0.3])
        segments = threshold_segments(scores, 0.5)
        assert segments == []

    def test_empty_input(self):
        assert threshold_segments(np.array([]), 0.5) == []

    def test_exactly_at_threshold(self):
        """Score exactly equal to threshold should be included (>=)."""
        scores = np.array([0.0, 0.5, 0.0])
        segments = threshold_segments(scores, 0.5)
        assert segments == [(1, 2)]


class TestApplyPadding:
    def test_basic_padding(self):
        segments = [(2.0, 3.0)]
        result = apply_padding(segments, pre_padding=1.0, post_padding=1.5, max_duration=10.0)
        assert result == [(1.0, 4.5)]

    def test_clamp_to_zero(self):
        segments = [(0.5, 1.0)]
        result = apply_padding(segments, pre_padding=1.0, post_padding=0.5, max_duration=10.0)
        assert result[0][0] == 0.0

    def test_clamp_to_max(self):
        segments = [(8.0, 9.5)]
        result = apply_padding(segments, pre_padding=0.5, post_padding=1.5, max_duration=10.0)
        assert result[0][1] == 10.0


class TestMergeOverlapping:
    def test_overlapping_segments(self):
        segments = [(0.0, 2.0), (1.5, 3.5)]
        result = merge_overlapping(segments)
        assert result == [(0.0, 3.5)]

    def test_nearly_adjacent(self):
        """Segments with gap < 0.5s should merge."""
        segments = [(0.0, 1.0), (1.3, 2.0)]
        result = merge_overlapping(segments, gap_threshold=0.5)
        assert result == [(0.0, 2.0)]

    def test_distant_segments(self):
        """Segments with gap >= 0.5s should not merge."""
        segments = [(0.0, 1.0), (2.0, 3.0)]
        result = merge_overlapping(segments, gap_threshold=0.5)
        assert result == [(0.0, 1.0), (2.0, 3.0)]

    def test_unsorted_input(self):
        segments = [(3.0, 4.0), (0.0, 1.0)]
        result = merge_overlapping(segments)
        assert result == [(0.0, 1.0), (3.0, 4.0)]

    def test_empty_input(self):
        assert merge_overlapping([]) == []

    def test_three_way_merge(self):
        segments = [(0.0, 1.0), (0.8, 2.0), (1.9, 3.0)]
        result = merge_overlapping(segments)
        assert result == [(0.0, 3.0)]


class TestFilterByDuration:
    def test_discard_short(self):
        segments = [(0.0, 0.3), (1.0, 3.0)]
        result = filter_by_duration(segments, min_duration=0.5, max_duration=15.0)
        assert result == [(1.0, 3.0)]

    def test_cap_long(self):
        segments = [(0.0, 20.0)]
        result = filter_by_duration(segments, min_duration=0.5, max_duration=10.0)
        assert result == [(0.0, 10.0)]

    def test_pass_through(self):
        segments = [(0.0, 5.0)]
        result = filter_by_duration(segments, min_duration=1.0, max_duration=15.0)
        assert result == [(0.0, 5.0)]


class TestExtractMoments:
    def test_single_peak(self):
        """A clear spike should produce one DetectedMoment."""
        scores = np.zeros(100)  # 10 seconds at 10 samples/sec
        scores[40:50] = 0.9  # Peak at 4-5 seconds
        moments = extract_moments(
            scores,
            source_path="test.mp4",
            threshold=0.5,
            pre_padding=0.5,
            post_padding=0.5,
            min_segment_duration=0.5,
            max_segment_duration=15.0,
        )
        assert len(moments) == 1
        m = moments[0]
        assert m.source_path == "test.mp4"
        assert m.start_time == pytest.approx(3.5, abs=0.11)
        assert m.end_time == pytest.approx(5.5, abs=0.11)
        assert m.peak_score == pytest.approx(0.9, abs=0.01)

    def test_two_peaks_ranked(self):
        """Two peaks should be returned ranked by peak score descending."""
        scores = np.zeros(100)
        scores[20:25] = 0.7  # Weaker peak
        scores[60:65] = 0.95  # Stronger peak
        moments = extract_moments(
            scores,
            source_path="test.mp4",
            threshold=0.5,
            pre_padding=0.0,
            post_padding=0.0,
            min_segment_duration=0.3,
            max_segment_duration=15.0,
        )
        assert len(moments) == 2
        assert moments[0].peak_score > moments[1].peak_score

    def test_no_peaks(self):
        scores = np.full(50, 0.2)
        moments = extract_moments(scores, "test.mp4", threshold=0.5)
        assert moments == []

    def test_empty_scores(self):
        moments = extract_moments(np.array([]), "test.mp4")
        assert moments == []

    def test_contributing_detectors_passed(self):
        scores = np.zeros(50)
        scores[20:30] = 0.8
        moments = extract_moments(
            scores,
            "test.mp4",
            threshold=0.5,
            contributing_detectors=["scene", "audio"],
        )
        assert len(moments) >= 1
        assert moments[0].contributing_detectors == ["scene", "audio"]

    def test_video_duration_clamps_padding(self):
        """Padding should not extend beyond video duration."""
        scores = np.zeros(20)  # 2 seconds
        scores[18:20] = 0.9  # Peak at end
        moments = extract_moments(
            scores,
            "test.mp4",
            threshold=0.5,
            pre_padding=0.5,
            post_padding=5.0,  # Would exceed 2s
            min_segment_duration=0.3,
            video_duration=2.0,
        )
        assert len(moments) >= 1
        assert moments[0].end_time == 2.0

    def test_short_segments_filtered(self):
        """Segments shorter than min_segment_duration should be discarded."""
        scores = np.zeros(100)
        scores[50] = 0.9  # Single sample spike = 0.1s
        moments = extract_moments(
            scores,
            "test.mp4",
            threshold=0.5,
            pre_padding=0.0,
            post_padding=0.0,
            min_segment_duration=1.0,
        )
        assert moments == []
