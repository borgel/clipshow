"""QThread worker for running the detection pipeline."""

from __future__ import annotations

import concurrent.futures
import threading
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

    Uses ThreadPoolExecutor to analyze multiple videos concurrently when
    max_workers > 1.  Each pool thread gets its own DetectionPipeline
    instance and emits per-file progress signals so the UI stays granular.

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
    warning = Signal(str)
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
        self._preview_lock = threading.Lock()

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

    def _maybe_emit_preview(self, path: str, duration: float, fraction: float) -> None:
        """Emit a frame preview if enough time has passed (thread-safe)."""
        now = time.monotonic()
        with self._preview_lock:
            if now - self._last_preview_time < _FRAME_PREVIEW_INTERVAL:
                return
            self._last_preview_time = now
        timestamp = fraction * duration
        img = self._extract_frame(path, timestamp)
        if img is not None:
            self.frame_preview.emit(img)

    def _analyze_one(self, path: str, duration: float) -> list[DetectedMoment]:
        """Analyze a single video — safe to call from any thread."""
        pipeline = DetectionPipeline(self.settings)

        def on_progress(p: float) -> None:
            self.progress.emit(path, p)
            self._maybe_emit_preview(path, duration, p)

        def on_warning(msg: str) -> None:
            self.warning.emit(msg)

        return pipeline.analyze_video(
            path,
            video_duration=duration,
            progress_callback=on_progress,
            cancel_flag=self._is_cancelled,
            warning_callback=on_warning,
        )

    def run(self) -> None:
        try:
            all_moments: list[DetectedMoment] = []
            max_workers = self.settings.resolved_max_workers
            max_workers = min(max_workers, len(self.video_paths))

            if max_workers <= 1 or len(self.video_paths) <= 1:
                self._run_sequential(all_moments)
            else:
                self._run_parallel(all_moments, max_workers)

            all_moments.sort(key=lambda m: m.peak_score, reverse=True)
            self.all_complete.emit(all_moments)

        except Exception as e:
            self.error.emit(str(e))

    def _run_sequential(self, all_moments: list[DetectedMoment]) -> None:
        for path, duration in self.video_paths:
            if self._cancelled:
                break
            basename = _Path(path).name
            self.status.emit(f"Analyzing {basename}...")
            moments = self._analyze_one(path, duration)
            all_moments.extend(moments)
            self.file_complete.emit(path)

    def _run_parallel(
        self, all_moments: list[DetectedMoment], max_workers: int
    ) -> None:
        n = len(self.video_paths)
        self.status.emit(f"Analyzing {n} videos in parallel...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path: dict[concurrent.futures.Future, str] = {}
            for path, duration in self.video_paths:
                future = executor.submit(self._analyze_one, path, duration)
                future_to_path[future] = path

            for future in concurrent.futures.as_completed(future_to_path):
                if self._cancelled:
                    for f in future_to_path:
                        f.cancel()
                    break
                path = future_to_path[future]
                try:
                    moments = future.result()
                except Exception as exc:
                    self.error.emit(f"{_Path(path).name}: {exc}")
                    continue
                all_moments.extend(moments)
                self.file_complete.emit(path)
