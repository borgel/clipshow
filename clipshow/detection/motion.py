"""Motion detection via OpenCV frame differencing."""

from __future__ import annotations

import cv2
import numpy as np

from clipshow.detection.base import Detector, DetectorResult


class MotionDetector(Detector):
    """Detects motion using OpenCV absdiff on decimated grayscale frames.

    Compares consecutive frames to measure pixel-level change.
    """

    name = "motion"

    def __init__(self, time_step: float = 0.1, decimate: int = 2):
        self._time_step = time_step
        self._decimate = decimate  # Read every Nth frame for speed

    def detect(
        self,
        video_path: str,
        progress_callback: callable | None = None,
        cancel_flag: callable | None = None,
    ) -> DetectorResult:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return DetectorResult(
                name=self.name,
                scores=np.array([]),
                time_step=self._time_step,
                source_path=video_path,
            )

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0 or frame_count <= 0:
            cap.release()
            return DetectorResult(
                name=self.name,
                scores=np.array([]),
                time_step=self._time_step,
                source_path=video_path,
            )

        duration = frame_count / fps
        num_samples = max(1, int(np.ceil(duration / self._time_step)))
        scores = np.zeros(num_samples, dtype=float)

        prev_gray = None
        frame_idx = 0

        while True:
            if cancel_flag and cancel_flag():
                break

            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % self._decimate == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Downsample for speed
                gray = cv2.resize(gray, (160, 120))

                if prev_gray is not None:
                    diff = cv2.absdiff(prev_gray, gray)
                    motion_score = float(diff.mean()) / 255.0

                    t = frame_idx / fps
                    idx = min(int(t / self._time_step), num_samples - 1)
                    scores[idx] = max(scores[idx], motion_score)

                prev_gray = gray

            frame_idx += 1

            if progress_callback and frame_count > 0:
                progress_callback(frame_idx / frame_count)

        cap.release()

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
