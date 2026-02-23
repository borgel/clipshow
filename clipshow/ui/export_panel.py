"""Export panel: output settings, summary, progress, and export control."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from clipshow.model.moments import HighlightSegment
from clipshow.model.project import ExportSettings
from clipshow.workers.export_worker import ExportWorker


class ExportPanel(QWidget):
    """Panel for configuring and running the video export."""

    export_complete = Signal(str)  # output_path
    export_error = Signal(str)

    def __init__(
        self,
        export_settings: ExportSettings | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.export_settings = export_settings or ExportSettings()
        self._segments: list[HighlightSegment] = []
        self._worker: ExportWorker | None = None
        self._setup_ui()
        self._connect_signals()
        self._load_settings()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Output path
        path_group = QGroupBox("Output")
        path_layout = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.browse_button = QPushButton("Browse...")
        path_layout.addWidget(self.path_edit)
        path_layout.addWidget(self.browse_button)
        path_group.setLayout(path_layout)
        layout.addWidget(path_group)

        # Settings
        settings_group = QGroupBox("Encoding Settings")
        settings_layout = QFormLayout()

        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 120)
        settings_layout.addRow("FPS:", self.fps_spin)

        self.bitrate_edit = QLineEdit()
        settings_layout.addRow("Bitrate:", self.bitrate_edit)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        # Summary
        self.summary_label = QLabel("No segments loaded")
        layout.addWidget(self.summary_label)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Buttons
        btn_layout = QHBoxLayout()
        self.export_button = QPushButton("Export")
        self.cancel_button = QPushButton("Cancel")
        self.export_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        btn_layout.addStretch()
        btn_layout.addWidget(self.export_button)
        btn_layout.addWidget(self.cancel_button)
        layout.addLayout(btn_layout)

        layout.addStretch()

    def _connect_signals(self) -> None:
        self.browse_button.clicked.connect(self._browse_output)
        self.export_button.clicked.connect(self.start_export)
        self.fps_spin.valueChanged.connect(self._on_fps_changed)
        self.bitrate_edit.textChanged.connect(self._on_bitrate_changed)
        self.path_edit.textChanged.connect(self._on_path_changed)

    def _load_settings(self) -> None:
        self.path_edit.setText(self.export_settings.output_path)
        self.fps_spin.setValue(int(self.export_settings.fps))
        self.bitrate_edit.setText(self.export_settings.bitrate)

    def _browse_output(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Highlight Reel",
            self.export_settings.output_path,
            "Video Files (*.mp4);;All Files (*)",
        )
        if path:
            self.path_edit.setText(path)

    def _on_fps_changed(self, value: int) -> None:
        self.export_settings.fps = float(value)

    def _on_bitrate_changed(self, text: str) -> None:
        self.export_settings.bitrate = text

    def _on_path_changed(self, text: str) -> None:
        self.export_settings.output_path = text

    def set_segments(self, segments: list[HighlightSegment]) -> None:
        """Load segments and update the summary."""
        self._segments = list(segments)
        included = [s for s in segments if s.included]
        total_duration = sum(s.duration for s in included)
        self.summary_label.setText(
            f"{len(included)} segments, {total_duration:.1f}s total"
        )
        self.export_button.setEnabled(len(included) > 0)

    def start_export(self) -> None:
        """Launch the export worker."""
        included = [s for s in self._segments if s.included]
        if not included:
            return

        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.export_button.setEnabled(False)
        self.cancel_button.setEnabled(True)

        self._worker = ExportWorker(included, self.export_settings)
        self._worker.progress.connect(self._on_progress)
        self._worker.complete.connect(self._on_complete)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, fraction: float) -> None:
        self.progress_bar.setValue(int(fraction * 100))

    def _on_complete(self, output_path: str) -> None:
        self.progress_bar.setValue(100)
        self.export_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self._worker = None
        self.export_complete.emit(output_path)

    def _on_error(self, message: str) -> None:
        self.progress_bar.setVisible(False)
        self.export_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self._worker = None
        self.export_error.emit(message)

    @property
    def is_exporting(self) -> bool:
        return self._worker is not None and self._worker.isRunning()
