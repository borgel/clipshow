"""Scene change detection via PySceneDetect."""

from __future__ import annotations

import numpy as np
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector
from scenedetect.stats_manager import StatsManager

from clipshow.detection.base import Detector, DetectorResult


class SceneDetector(Detector):
    """Detects scene changes using PySceneDetect's ContentDetector.

    Produces score spikes at cuts and transitions based on HSV-weighted
    frame differencing.
    """

    name = "scene"

    def __init__(self, threshold: float = 27.0, time_step: float = 0.1):
        self._threshold = threshold
        self._time_step = time_step

    def detect(
        self,
        video_path: str,
        progress_callback: callable | None = None,
        cancel_flag: callable | None = None,
    ) -> DetectorResult:
        video = open_video(video_path)
        stats = StatsManager()
        scene_manager = SceneManager(stats_manager=stats)
        scene_manager.add_detector(ContentDetector(threshold=self._threshold))

        scene_manager.detect_scenes(video=video, show_progress=False)

        if progress_callback:
            progress_callback(0.5)  # Scene analysis done, score extraction next

        fps = video.frame_rate
        frame_count = video.duration.get_frames()

        if fps <= 0 or frame_count <= 0:
            return DetectorResult(
                name=self.name,
                scores=np.array([]),
                time_step=self._time_step,
                source_path=video_path,
            )

        duration = frame_count / fps
        num_samples = max(1, int(np.ceil(duration / self._time_step)))
        scores = np.zeros(num_samples, dtype=float)

        # Extract per-frame content_val from the stats manager.
        # ContentDetector stores the combined score as "content_val".
        metric_keys = stats.metric_keys
        if metric_keys and ContentDetector.FRAME_SCORE_KEY in metric_keys:
            content_key = ContentDetector.FRAME_SCORE_KEY

            for frame_num in range(frame_count):
                if stats.metrics_exist(frame_num, [content_key]):
                    vals = stats.get_metrics(frame_num, [content_key])
                    val = vals[0] if vals else 0.0
                    if val is not None:
                        t = frame_num / fps
                        idx = min(int(t / self._time_step), num_samples - 1)
                        scores[idx] = max(scores[idx], float(val))

        # Normalize to [0, 1]
        max_val = scores.max()
        if max_val > 0:
            scores = scores / max_val

        if progress_callback:
            progress_callback(1.0)

        return DetectorResult(
            name=self.name,
            scores=scores,
            time_step=self._time_step,
            source_path=video_path,
        )
