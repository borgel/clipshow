"""Analyze panel: detector settings, progress bars, and analysis control."""

from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from clipshow.config import Settings
from clipshow.model.project import Project
from clipshow.ui.prompt_editor import PromptEditor
from clipshow.workers.analysis_worker import AnalysisWorker

# Slider scale: sliders are 0-100 integers, mapped to 0.0-1.0 floats
SLIDER_SCALE = 100

# Status markers for the file list
_PENDING = "\u2500"  # dash
_ANALYZING = "\u25B6"  # play triangle
_COMPLETE = "\u2714"  # checkmark


class AnalyzePanel(QWidget):
    """Panel for configuring detectors and running analysis."""

    analysis_complete = Signal(list)  # list[DetectedMoment]
    analysis_started = Signal()

    def __init__(
        self,
        project: Project,
        settings: Settings | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.project = project
        self.settings = settings or Settings()
        self._worker: AnalysisWorker | None = None
        self._total_files: int = 0
        self._completed_files: int = 0
        self._has_results: bool = False
        self._analysis_start_time: float = 0.0
        self._total_video_duration: float = 0.0
        self._source_paths: list[str] = []
        self._setup_ui()
        self._connect_signals()
        self._load_settings()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(splitter)

        # --- Top: settings in a scroll area ---
        settings_widget = QWidget()
        settings_layout = QVBoxLayout(settings_widget)
        settings_layout.setContentsMargins(4, 4, 4, 4)

        # Detector weights group
        weights_group = QGroupBox("Detector Weights")
        weights_layout = QFormLayout()
        weights_layout.setContentsMargins(6, 6, 6, 6)
        weights_layout.setVerticalSpacing(4)

        self.scene_check = QCheckBox()
        self.scene_slider = QSlider(Qt.Orientation.Horizontal)
        self.scene_slider.setRange(0, SLIDER_SCALE)
        self.scene_label = QLabel()
        row = QHBoxLayout()
        row.addWidget(self.scene_check)
        row.addWidget(self.scene_slider)
        row.addWidget(self.scene_label)
        weights_layout.addRow("Scene:", row)

        self.audio_check = QCheckBox()
        self.audio_slider = QSlider(Qt.Orientation.Horizontal)
        self.audio_slider.setRange(0, SLIDER_SCALE)
        self.audio_label = QLabel()
        row = QHBoxLayout()
        row.addWidget(self.audio_check)
        row.addWidget(self.audio_slider)
        row.addWidget(self.audio_label)
        weights_layout.addRow("Audio:", row)

        self.motion_check = QCheckBox()
        self.motion_slider = QSlider(Qt.Orientation.Horizontal)
        self.motion_slider.setRange(0, SLIDER_SCALE)
        self.motion_label = QLabel()
        row = QHBoxLayout()
        row.addWidget(self.motion_check)
        row.addWidget(self.motion_slider)
        row.addWidget(self.motion_label)
        weights_layout.addRow("Motion:", row)

        self.semantic_check = QCheckBox()
        self.semantic_slider = QSlider(Qt.Orientation.Horizontal)
        self.semantic_slider.setRange(0, SLIDER_SCALE)
        self.semantic_label = QLabel()
        self.edit_prompts_button = QPushButton("Edit Prompts…")
        row = QHBoxLayout()
        row.addWidget(self.semantic_check)
        row.addWidget(self.semantic_slider)
        row.addWidget(self.semantic_label)
        row.addWidget(self.edit_prompts_button)
        weights_layout.addRow("Semantic:", row)

        self.emotion_check = QCheckBox()
        self.emotion_slider = QSlider(Qt.Orientation.Horizontal)
        self.emotion_slider.setRange(0, SLIDER_SCALE)
        self.emotion_label = QLabel()
        row = QHBoxLayout()
        row.addWidget(self.emotion_check)
        row.addWidget(self.emotion_slider)
        row.addWidget(self.emotion_label)
        weights_layout.addRow("Emotion:", row)

        weights_help = QLabel(
            "Control how much each detector contributes to the highlight score. "
            "Higher weight = more influence."
        )
        weights_help.setWordWrap(True)
        weights_help.setStyleSheet("color: gray;")
        weights_layout.addRow(weights_help)

        weights_group.setLayout(weights_layout)
        settings_layout.addWidget(weights_group)

        # Threshold slider
        threshold_group = QGroupBox("Threshold")
        threshold_layout = QFormLayout()
        threshold_layout.setContentsMargins(6, 6, 6, 6)
        threshold_layout.setVerticalSpacing(4)

        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, SLIDER_SCALE)
        self.threshold_label = QLabel()
        row = QHBoxLayout()
        row.addWidget(self.threshold_slider)
        row.addWidget(self.threshold_label)
        threshold_layout.addRow("Score threshold:", row)

        threshold_help = QLabel(
            "Minimum combined score for a moment to become a highlight. "
            "Lower = more highlights, higher = only the best."
        )
        threshold_help.setWordWrap(True)
        threshold_help.setStyleSheet("color: gray;")
        threshold_layout.addRow(threshold_help)

        threshold_group.setLayout(threshold_layout)
        settings_layout.addWidget(threshold_group)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(settings_widget)
        splitter.addWidget(scroll)

        # --- Bottom: progress area with file list ---
        progress_widget = QWidget()
        progress_layout = QVBoxLayout(progress_widget)
        progress_layout.setContentsMargins(4, 4, 4, 4)

        # Status and progress bar
        self.status_label = QLabel("")
        progress_layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.hide()
        progress_layout.addWidget(self.progress_bar)

        # File status list
        self.file_list = QListWidget()
        self.file_list.setAlternatingRowColors(True)
        progress_layout.addWidget(self.file_list, stretch=1)

        # Buttons
        btn_layout = QHBoxLayout()
        self.analyze_button = QPushButton("Analyze All")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        btn_layout.addStretch()
        btn_layout.addWidget(self.analyze_button)
        btn_layout.addWidget(self.cancel_button)
        progress_layout.addLayout(btn_layout)

        splitter.addWidget(progress_widget)

        # Give both halves equal starting weight
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

    def _connect_signals(self) -> None:
        self.scene_check.toggled.connect(lambda on: self._on_check_toggled("scene", on))
        self.audio_check.toggled.connect(lambda on: self._on_check_toggled("audio", on))
        self.motion_check.toggled.connect(lambda on: self._on_check_toggled("motion", on))
        self.semantic_check.toggled.connect(
            lambda on: self._on_check_toggled("semantic", on)
        )
        self.emotion_check.toggled.connect(
            lambda on: self._on_check_toggled("emotion", on)
        )

        self.scene_slider.valueChanged.connect(
            lambda v: self._on_slider_changed("scene", v)
        )
        self.audio_slider.valueChanged.connect(
            lambda v: self._on_slider_changed("audio", v)
        )
        self.motion_slider.valueChanged.connect(
            lambda v: self._on_slider_changed("motion", v)
        )
        self.semantic_slider.valueChanged.connect(
            lambda v: self._on_slider_changed("semantic", v)
        )
        self.emotion_slider.valueChanged.connect(
            lambda v: self._on_slider_changed("emotion", v)
        )
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)

        self.edit_prompts_button.clicked.connect(self._open_prompt_editor)
        self.analyze_button.clicked.connect(self.start_analysis)
        self.cancel_button.clicked.connect(self.cancel_analysis)

    def _load_settings(self) -> None:
        """Populate UI from settings."""
        self.scene_slider.setValue(int(self.settings.scene_weight * SLIDER_SCALE))
        self.audio_slider.setValue(int(self.settings.audio_weight * SLIDER_SCALE))
        self.motion_slider.setValue(int(self.settings.motion_weight * SLIDER_SCALE))
        self.semantic_slider.setValue(int(self.settings.semantic_weight * SLIDER_SCALE))
        self.emotion_slider.setValue(int(self.settings.emotion_weight * SLIDER_SCALE))
        self.threshold_slider.setValue(
            int(self.settings.score_threshold * SLIDER_SCALE)
        )

        self.scene_check.setChecked(self.settings.scene_weight > 0)
        self.audio_check.setChecked(self.settings.audio_weight > 0)
        self.motion_check.setChecked(self.settings.motion_weight > 0)
        self.semantic_check.setChecked(self.settings.semantic_weight > 0)
        self.emotion_check.setChecked(self.settings.emotion_weight > 0)

    def _on_check_toggled(self, detector: str, enabled: bool) -> None:
        slider = getattr(self, f"{detector}_slider")
        slider.setEnabled(enabled)
        if not enabled:
            slider.setValue(0)

    def _on_slider_changed(self, detector: str, value: int) -> None:
        weight = value / SLIDER_SCALE
        setattr(self.settings, f"{detector}_weight", weight)
        label = getattr(self, f"{detector}_label")
        label.setText(f"{value}%")

    def _on_threshold_changed(self, value: int) -> None:
        self.settings.score_threshold = value / SLIDER_SCALE
        self.threshold_label.setText(f"{value}%")

    def _open_prompt_editor(self) -> None:
        """Open a dialog with the PromptEditor widget."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Edit Semantic Prompts")
        dlg.setMinimumSize(400, 300)
        layout = QVBoxLayout(dlg)

        editor = PromptEditor(self.settings.semantic_prompts)
        layout.addWidget(editor)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.settings.semantic_prompts = editor.prompts

    def _populate_file_list(self) -> None:
        """Fill the file list with source filenames and pending status."""
        self.file_list.clear()
        for path in self._source_paths:
            name = Path(path).name
            item = QListWidgetItem(f"{_PENDING}  {name}")
            item.setForeground(Qt.GlobalColor.gray)
            self.file_list.addItem(item)

    def _update_file_status(self, index: int, marker: str) -> None:
        """Update a single file's status marker and color in the list."""
        if index < 0 or index >= self.file_list.count():
            return
        item = self.file_list.item(index)
        name = Path(self._source_paths[index]).name
        item.setText(f"{marker}  {name}")
        if marker == _COMPLETE:
            item.setForeground(Qt.GlobalColor.darkGreen)
        elif marker == _ANALYZING:
            item.setForeground(Qt.GlobalColor.white)
        else:
            item.setForeground(Qt.GlobalColor.gray)

    def start_analysis(self) -> None:
        """Launch the analysis worker thread."""
        if not self.project.sources:
            return

        self._total_files = len(self.project.sources)
        self._completed_files = 0
        self._analysis_start_time = time.monotonic()
        self._total_video_duration = sum(s.duration for s in self.project.sources)
        self._source_paths = [s.path for s in self.project.sources]

        self._populate_file_list()

        video_paths = [(s.path, s.duration) for s in self.project.sources]
        self._worker = AnalysisWorker(video_paths, self.settings)
        self._worker.progress.connect(self._on_progress)
        self._worker.file_complete.connect(self._on_file_complete)
        self._worker.all_complete.connect(self._on_all_complete)
        self._worker.error.connect(self._on_error)
        self._worker.status.connect(self._on_status)

        n = self._total_files
        self.status_label.setText(f"Analyzing {n} video{'s' if n != 1 else ''}...")
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.analyze_button.setEnabled(False)
        self.cancel_button.setEnabled(True)

        # Mark first file as analyzing
        self._update_file_status(0, _ANALYZING)

        self._worker.start()
        self.analysis_started.emit()

    def cancel_analysis(self) -> None:
        """Request cancellation of the running worker."""
        if self._worker:
            self._worker.cancel()
            self.status_label.setText("Cancelling...")

    def _on_status(self, message: str) -> None:
        self.status_label.setText(message)

    @staticmethod
    def _format_eta(seconds: float) -> str:
        """Format seconds into a human-readable ETA string."""
        if seconds < 60:
            return f"~{int(seconds)}s remaining"
        minutes = int(seconds) // 60
        secs = int(seconds) % 60
        return f"~{minutes}m {secs:02d}s remaining"

    def _on_progress(self, source_path: str, fraction: float) -> None:
        # Overall progress = (completed files + current file fraction) / total files
        overall = (self._completed_files + fraction) / self._total_files
        self.progress_bar.setValue(int(overall * 100))
        basename = Path(source_path).name

        # Compute rate and ETA
        elapsed = time.monotonic() - self._analysis_start_time
        rate_str = ""
        eta_str = ""
        if elapsed > 1.0 and overall > 0.01:
            processed_duration = self._total_video_duration * overall
            rate = processed_duration / elapsed
            rate_str = f"{rate:.1f}x realtime"
            eta = elapsed * (1 - overall) / overall
            eta_str = self._format_eta(eta)

        parts = [
            f"Analyzing {basename} "
            f"({self._completed_files + 1} of {self._total_files})"
        ]
        if rate_str:
            parts.append(f"\u2014 {rate_str}, {eta_str}")
        self.status_label.setText(" ".join(parts))

    def _on_file_complete(self, source_path: str) -> None:
        # Mark completed file
        self._update_file_status(self._completed_files, _COMPLETE)
        self._completed_files += 1

        # Mark next file as analyzing (if any)
        if self._completed_files < self._total_files:
            self._update_file_status(self._completed_files, _ANALYZING)

        basename = Path(source_path).name
        self.status_label.setText(
            f"Completed {basename} ({self._completed_files} of {self._total_files})"
        )

    def _on_all_complete(self, moments: list) -> None:
        n = self._total_files
        count = len(moments)
        self.status_label.setText(
            f"Analysis complete \u2014 found {count} highlight{'s' if count != 1 else ''} "
            f"in {n} video{'s' if n != 1 else ''}"
        )
        self.progress_bar.hide()
        self.analyze_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self._worker = None
        self._has_results = True
        self.analysis_complete.emit(moments)

    def _on_error(self, message: str) -> None:
        self.status_label.setText(f"Error: {message}")
        self.progress_bar.hide()
        self.analyze_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self._worker = None

    @property
    def has_results(self) -> bool:
        return self._has_results

    @property
    def is_analyzing(self) -> bool:
        return self._worker is not None and self._worker.isRunning()
