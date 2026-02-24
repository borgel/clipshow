"""Detection pipeline: runs all enabled detectors and delegates to scoring."""

from __future__ import annotations

import concurrent.futures
from dataclasses import asdict

import numpy as np

from clipshow.config import Settings
from clipshow.detection.audio import AudioDetector
from clipshow.detection.base import Detector, DetectorResult
from clipshow.detection.motion import MotionDetector
from clipshow.detection.scene import SceneDetector
from clipshow.detection.scoring import (
    extract_moments,
    resample_scores,
    weighted_combine,
)
from clipshow.model.moments import DetectedMoment

DETECTOR_CLASSES: dict[str, type[Detector]] = {
    "scene": SceneDetector,
    "audio": AudioDetector,
    "motion": MotionDetector,
}


def _get_optional_detector(name: str) -> type[Detector] | None:
    """Lazy-load optional detectors (semantic, emotion) on demand."""
    if name == "semantic":
        try:
            from clipshow.detection.semantic import SemanticDetector
            return SemanticDetector
        except ImportError:
            return None
    elif name == "emotion":
        try:
            from clipshow.detection.emotion import EmotionDetector
            return EmotionDetector
        except ImportError:
            return None
    return None


def _analyze_single_video(settings_dict: dict, video_path: str, video_duration: float):
    """Top-level function for ProcessPoolExecutor (must be picklable)."""
    settings = Settings(**settings_dict)
    pipeline = DetectionPipeline(settings)
    return pipeline.analyze_video(video_path, video_duration)


class DetectionPipeline:
    """Orchestrates running detectors and extracting highlight moments."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()

    def _build_detectors(
        self, warning_callback: callable | None = None
    ) -> list[tuple[str, Detector, float]]:
        """Return (name, detector_instance, weight) for enabled detectors."""
        result = []
        for name, weight in self.settings.detector_weights.items():
            if weight <= 0:
                continue
            if name in DETECTOR_CLASSES:
                result.append((name, DETECTOR_CLASSES[name](), weight))
            else:
                # Try lazy-loaded optional detectors
                cls = _get_optional_detector(name)
                if cls is not None:
                    kwargs = {}
                    if name == "semantic":
                        kwargs["prompts"] = self.settings.semantic_prompts
                        kwargs["negative_prompts"] = self.settings.semantic_negative_prompts
                    result.append((name, cls(**kwargs), weight))
                elif warning_callback:
                    warning_callback(
                        f"{name.title()} detector skipped: missing dependencies. "
                        f'Install with: uv sync --extra {name}'
                    )
        return result

    def analyze_video(
        self,
        video_path: str,
        video_duration: float,
        progress_callback: callable | None = None,
        cancel_flag: callable | None = None,
        warning_callback: callable | None = None,
    ) -> list[DetectedMoment]:
        """Run all enabled detectors on a single video and extract moments.

        Args:
            video_path: Path to the video file.
            video_duration: Total duration in seconds (from ffprobe).
            progress_callback: Optional callable(float) for overall progress (0-1).
            cancel_flag: Optional callable() -> bool for cancellation.
            warning_callback: Optional callable(str) for non-fatal warnings.

        Returns:
            List of DetectedMoment sorted by peak_score descending.
        """
        detectors = self._build_detectors(warning_callback=warning_callback)
        if not detectors:
            return []

        results: list[DetectorResult] = []
        weights: list[float] = []
        detector_names: list[str] = []

        for i, (name, detector, weight) in enumerate(detectors):
            if cancel_flag and cancel_flag():
                break

            def per_detector_progress(p: float) -> None:
                if progress_callback:
                    overall = (i + p) / len(detectors)
                    progress_callback(overall)

            try:
                result = detector.detect(
                    video_path,
                    progress_callback=per_detector_progress,
                    cancel_flag=cancel_flag,
                )
            except RuntimeError as exc:
                # Optional detector failed to load (missing dependency) — skip
                if warning_callback:
                    warning_callback(f"{name.title()} detector skipped: {exc}")
                continue

            if len(result.scores) > 0:
                results.append(result)
                weights.append(weight)
                detector_names.append(name)

        if not results:
            if progress_callback:
                progress_callback(1.0)
            return []

        # Align all resampled arrays to the same length
        resampled_arrays = []
        for r in results:
            resampled_arrays.append(resample_scores(r.scores, r.time_step))

        max_len = max(len(a) for a in resampled_arrays)
        aligned = []
        for a in resampled_arrays:
            if len(a) < max_len:
                padded = np.zeros(max_len)
                padded[: len(a)] = a
                aligned.append(padded)
            else:
                aligned.append(a[:max_len])

        combined = weighted_combine(aligned, weights)

        moments = extract_moments(
            combined,
            source_path=video_path,
            threshold=self.settings.score_threshold,
            pre_padding=self.settings.pre_padding_sec,
            post_padding=self.settings.post_padding_sec,
            min_segment_duration=self.settings.min_segment_duration_sec,
            max_segment_duration=self.settings.max_segment_duration_sec,
            video_duration=video_duration,
            contributing_detectors=detector_names,
        )

        if progress_callback:
            progress_callback(1.0)

        return moments

    def analyze_all(
        self,
        video_paths: list[tuple[str, float]],
        progress_callback: callable | None = None,
        cancel_flag: callable | None = None,
    ) -> list[DetectedMoment]:
        """Run detection on multiple videos.

        Args:
            video_paths: List of (path, duration) tuples.
            progress_callback: Overall progress (0-1).
            cancel_flag: Cancellation flag.

        Returns:
            All detected moments across all videos, sorted by peak_score.
        """
        workers = self.settings.resolved_max_workers
        if workers == 1 or len(video_paths) <= 1:
            return self._analyze_all_sequential(
                video_paths, progress_callback, cancel_flag
            )
        return self._analyze_all_parallel(
            video_paths, workers, progress_callback, cancel_flag
        )

    def _analyze_all_sequential(
        self,
        video_paths: list[tuple[str, float]],
        progress_callback: callable | None = None,
        cancel_flag: callable | None = None,
    ) -> list[DetectedMoment]:
        """Sequential analysis — preserves per-detector progress callbacks."""
        all_moments: list[DetectedMoment] = []

        for i, (path, duration) in enumerate(video_paths):
            if cancel_flag and cancel_flag():
                break

            def per_video_progress(p: float) -> None:
                if progress_callback:
                    overall = (i + p) / len(video_paths)
                    progress_callback(overall)

            moments = self.analyze_video(
                path,
                video_duration=duration,
                progress_callback=per_video_progress,
                cancel_flag=cancel_flag,
            )
            all_moments.extend(moments)

        all_moments.sort(key=lambda m: m.peak_score, reverse=True)

        if progress_callback:
            progress_callback(1.0)

        return all_moments

    def _analyze_all_parallel(
        self,
        video_paths: list[tuple[str, float]],
        workers: int,
        progress_callback: callable | None = None,
        cancel_flag: callable | None = None,
    ) -> list[DetectedMoment]:
        """Parallel analysis using ProcessPoolExecutor."""
        all_moments: list[DetectedMoment] = []
        settings_dict = asdict(self.settings)
        total = len(video_paths)
        completed = 0

        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_path = {
                executor.submit(
                    _analyze_single_video, settings_dict, path, duration
                ): path
                for path, duration in video_paths
            }

            for future in concurrent.futures.as_completed(future_to_path):
                if cancel_flag and cancel_flag():
                    # Cancel remaining futures
                    for f in future_to_path:
                        f.cancel()
                    break

                moments = future.result()
                all_moments.extend(moments)
                completed += 1

                if progress_callback:
                    progress_callback(completed / total)

        all_moments.sort(key=lambda m: m.peak_score, reverse=True)

        if progress_callback:
            progress_callback(1.0)

        return all_moments
