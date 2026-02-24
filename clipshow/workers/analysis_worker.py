"""QThread worker for running the detection pipeline."""

from __future__ import annotations

from pathlib import Path as _Path

from PySide6.QtCore import QThread, Signal

from clipshow.config import Settings
from clipshow.detection.pipeline import DetectionPipeline
from clipshow.model.moments import DetectedMoment


class AnalysisWorker(QThread):
    """Runs detection pipeline in a background thread.

    Signals:
        progress: (source_path, fraction 0-1)
        file_complete: source_path
        all_complete: list of all DetectedMoment
        error: error message string
    """

    progress = Signal(str, float)
    file_complete = Signal(str)
    all_complete = Signal(list)
    error = Signal(str)
    status = Signal(str)

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

    def cancel(self) -> None:
        self._cancelled = True

    def _is_cancelled(self) -> bool:
        return self._cancelled

    def run(self) -> None:
        try:
            pipeline = DetectionPipeline(self.settings)
            all_moments: list[DetectedMoment] = []

            for i, (path, duration) in enumerate(self.video_paths):
                if self._cancelled:
                    break

                basename = _Path(path).name
                self.status.emit(f"Analyzing {basename}...")

                def on_progress(p: float, _path=path) -> None:
                    self.progress.emit(_path, p)

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
