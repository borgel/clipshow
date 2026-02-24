"""QThread worker for running the detection pipeline."""

from __future__ import annotations

import time
from pathlib import Path as _Path

import cv2
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from clipshow.config import Settings
from clipshow.detection.pipeline import DetectionPipeline
from clipshow.model.moments import DetectedMoment

# Throttle frame preview to at most once per second
_FRAME_PREVIEW_INTERVAL = 1.0
_PREVIEW_WIDTH = 320
_PREVIEW_HEIGHT = 180


class AnalysisWorker(QThread):
    """Runs detection pipeline in a background thread.

    Signals:
        progress: (source_path, fraction 0-1)
        file_complete: source_path
        all_complete: list of all DetectedMoment
        error: error message string
        status: status text
        frame_preview: QImage thumbnail from current analysis position
    """

    progress = Signal(str, float)
    file_complete = Signal(str)
    all_complete = Signal(list)
    error = Signal(str)
    status = Signal(str)
    frame_preview = Signal(object)  # QImage

    def __init__(
        self,
        video_paths: list[tuple[str, float]],
        settings: Settings | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.video_paths = video_paths
        self.settings = settings or Settings()
        self._cancelled = False
        self._last_preview_time = 0.0

    def cancel(self) -> None:
        self._cancelled = True

    def _is_cancelled(self) -> bool:
        return self._cancelled

    def _extract_frame(self, path: str, timestamp: float) -> QImage | None:
        """Extract a single frame from video at the given timestamp."""
        cap = cv2.VideoCapture(path)
        try:
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ret, frame = cap.read()
            if not ret or frame is None:
                return None
            frame = cv2.resize(frame, (_PREVIEW_WIDTH, _PREVIEW_HEIGHT))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            return img.copy()  # must copy — underlying data goes out of scope
        except Exception:
            return None
        finally:
            cap.release()

    def run(self) -> None:
        try:
            pipeline = DetectionPipeline(self.settings)
            all_moments: list[DetectedMoment] = []

            for i, (path, duration) in enumerate(self.video_paths):
                if self._cancelled:
                    break

                basename = _Path(path).name
                self.status.emit(f"Analyzing {basename}...")

                def on_progress(p: float, _path=path, _dur=duration) -> None:
                    self.progress.emit(_path, p)
                    # Throttled frame preview
                    now = time.monotonic()
                    if now - self._last_preview_time >= _FRAME_PREVIEW_INTERVAL:
                        self._last_preview_time = now
                        timestamp = p * _dur
                        img = self._extract_frame(_path, timestamp)
                        if img is not None:
                            self.frame_preview.emit(img)

                moments = pipeline.analyze_video(
                    path,
                    video_duration=duration,
                    progress_callback=on_progress,
                    cancel_flag=self._is_cancelled,
                )
                all_moments.extend(moments)
                self.file_complete.emit(path)

            all_moments.sort(key=lambda m: m.peak_score, reverse=True)
            self.all_complete.emit(all_moments)

        except Exception as e:
            self.error.emit(str(e))
