"""QThread worker for video export/encoding."""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from clipshow.export.assembler import assemble_highlights
from clipshow.model.moments import HighlightSegment
from clipshow.model.project import ExportSettings


class ExportWorker(QThread):
    """Runs video assembly in a background thread.

    Signals:
        progress: fraction 0-1
        complete: output_path
        error: error message
    """

    progress = Signal(float)
    complete = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        segments: list[HighlightSegment],
        export_settings: ExportSettings,
        parent=None,
    ):
        super().__init__(parent)
        self.segments = segments
        self.export_settings = export_settings

    def run(self) -> None:
        try:
            output = assemble_highlights(
                self.segments,
                output_path=self.export_settings.output_path,
                codec=self.export_settings.codec,
                fps=self.export_settings.fps,
                bitrate=self.export_settings.bitrate,
                progress_callback=lambda p: self.progress.emit(p),
            )
            self.complete.emit(output)
        except Exception as e:
            self.error.emit(str(e))
