"""Weighted score combination, thresholding, and segment extraction."""

import numpy as np

from clipshow.model.moments import DetectedMoment

# Common time base: 10 samples per second
COMMON_SAMPLE_RATE = 10.0
COMMON_TIME_STEP = 1.0 / COMMON_SAMPLE_RATE


def resample_scores(scores: np.ndarray, source_time_step: float) -> np.ndarray:
    """Resample a score array to the common 10 samples/sec time base.

    Uses linear interpolation.
    """
    if len(scores) == 0:
        return np.array([], dtype=float)

    source_duration = len(scores) * source_time_step
    num_output = max(1, int(np.ceil(source_duration * COMMON_SAMPLE_RATE)))

    source_times = np.arange(len(scores)) * source_time_step
    target_times = np.arange(num_output) * COMMON_TIME_STEP

    return np.interp(target_times, source_times, scores)


def weighted_combine(
    score_arrays: list[np.ndarray],
    weights: list[float],
) -> np.ndarray:
    """Combine multiple score arrays with weights, normalize to [0, 1].

    All arrays must be the same length (pre-resampled to common time base).
    Weights do not need to sum to 1 — they are normalized internally.
    """
    if not score_arrays or not weights:
        return np.array([], dtype=float)

    total_weight = sum(weights)
    if total_weight == 0:
        return np.zeros(len(score_arrays[0]), dtype=float)

    combined = np.zeros(len(score_arrays[0]), dtype=float)
    for scores, weight in zip(score_arrays, weights):
        combined += scores * (weight / total_weight)

    # Renormalize to [0, 1]
    max_val = combined.max()
    if max_val > 0:
        combined = combined / max_val
    return combined


def threshold_segments(
    scores: np.ndarray,
    threshold: float,
) -> list[tuple[int, int]]:
    """Find contiguous runs of scores >= threshold.

    Returns list of (start_idx, end_idx) pairs (end is exclusive).
    """
    if len(scores) == 0:
        return []

    above = scores >= threshold
    # Find transitions
    diff = np.diff(above.astype(int))
    starts = list(np.where(diff == 1)[0] + 1)
    ends = list(np.where(diff == -1)[0] + 1)

    # Handle edge cases
    if above[0]:
        starts.insert(0, 0)
    if above[-1]:
        ends.append(len(scores))

    return list(zip(starts, ends))


def apply_padding(
    segments: list[tuple[float, float]],
    pre_padding: float,
    post_padding: float,
    max_duration: float,
) -> list[tuple[float, float]]:
    """Apply pre/post padding to time-based segments, clamped to [0, max_duration]."""
    padded = []
    for start, end in segments:
        new_start = max(0.0, start - pre_padding)
        new_end = min(max_duration, end + post_padding)
        padded.append((new_start, new_end))
    return padded


def merge_overlapping(
    segments: list[tuple[float, float]],
    gap_threshold: float = 0.5,
) -> list[tuple[float, float]]:
    """Merge segments that overlap or are nearly adjacent (gap < gap_threshold)."""
    if not segments:
        return []

    # Sort by start time
    sorted_segs = sorted(segments, key=lambda s: s[0])
    merged = [sorted_segs[0]]

    for start, end in sorted_segs[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + gap_threshold:
            # Merge
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return merged


def filter_by_duration(
    segments: list[tuple[float, float]],
    min_duration: float,
    max_duration: float,
) -> list[tuple[float, float]]:
    """Discard segments shorter than min_duration, cap at max_duration."""
    filtered = []
    for start, end in segments:
        duration = end - start
        if duration < min_duration:
            continue
        if duration > max_duration:
            end = start + max_duration
        filtered.append((start, end))
    return filtered


def extract_moments(
    scores: np.ndarray,
    source_path: str,
    threshold: float = 0.5,
    pre_padding: float = 1.0,
    post_padding: float = 1.5,
    min_segment_duration: float = 1.0,
    max_segment_duration: float = 15.0,
    video_duration: float | None = None,
    contributing_detectors: list[str] | None = None,
) -> list[DetectedMoment]:
    """Full pipeline: threshold → pad → merge → filter → rank.

    Args:
        scores: Combined score array at COMMON_SAMPLE_RATE.
        source_path: Path to the source video.
        threshold: Minimum score to consider interesting.
        pre_padding: Seconds to add before each segment.
        post_padding: Seconds to add after each segment.
        min_segment_duration: Discard shorter segments.
        max_segment_duration: Cap segment length.
        video_duration: Total video duration for boundary clamping.
        contributing_detectors: Names of detectors that contributed.

    Returns:
        List of DetectedMoment sorted by peak_score descending.
    """
    if len(scores) == 0:
        return []

    if video_duration is None:
        video_duration = len(scores) * COMMON_TIME_STEP

    # 1. Threshold to find candidate index ranges
    index_segments = threshold_segments(scores, threshold)

    # 2. Convert index ranges to time ranges
    time_segments = [
        (start * COMMON_TIME_STEP, end * COMMON_TIME_STEP)
        for start, end in index_segments
    ]

    # 3. Apply padding
    time_segments = apply_padding(time_segments, pre_padding, post_padding, video_duration)

    # 4. Merge overlapping
    time_segments = merge_overlapping(time_segments)

    # 5. Filter by duration
    time_segments = filter_by_duration(
        time_segments, min_segment_duration, max_segment_duration
    )

    # 6. Build DetectedMoments with peak/mean scores
    moments = []
    for start, end in time_segments:
        start_idx = max(0, int(start * COMMON_SAMPLE_RATE))
        end_idx = min(len(scores), int(end * COMMON_SAMPLE_RATE))
        if end_idx <= start_idx:
            continue
        segment_scores = scores[start_idx:end_idx]
        moments.append(
            DetectedMoment(
                source_path=source_path,
                start_time=start,
                end_time=end,
                peak_score=float(segment_scores.max()),
                mean_score=float(segment_scores.mean()),
                contributing_detectors=contributing_detectors or [],
            )
        )

    # 7. Rank by peak score descending
    moments.sort(key=lambda m: m.peak_score, reverse=True)
    return moments
