"""Analyze panel: detector settings, progress bars, and analysis control."""

from __future__ import annotations

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
    QProgressBar,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from clipshow.config import Settings
from clipshow.model.project import Project
from clipshow.ui.prompt_editor import PromptEditor
from clipshow.workers.analysis_worker import AnalysisWorker

# Slider scale: sliders are 0-100 integers, mapped to 0.0-1.0 floats
SLIDER_SCALE = 100


class AnalyzePanel(QWidget):
    """Panel for configuring detectors and running analysis."""

    analysis_complete = Signal(list)  # list[DetectedMoment]

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
        self._setup_ui()
        self._connect_signals()
        self._load_settings()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Detector weights group
        weights_group = QGroupBox("Detector Weights")
        weights_layout = QFormLayout()

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
        layout.addWidget(weights_group)

        # Threshold slider
        threshold_group = QGroupBox("Threshold")
        threshold_layout = QFormLayout()

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
        layout.addWidget(threshold_group)

        # Status and progress section
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        # Buttons
        btn_layout = QHBoxLayout()
        self.analyze_button = QPushButton("Analyze All")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        btn_layout.addStretch()
        btn_layout.addWidget(self.analyze_button)
        btn_layout.addWidget(self.cancel_button)
        layout.addLayout(btn_layout)

        layout.addStretch()

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

    def start_analysis(self) -> None:
        """Launch the analysis worker thread."""
        if not self.project.sources:
            return

        self._total_files = len(self.project.sources)
        self._completed_files = 0

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
        self._worker.start()

    def cancel_analysis(self) -> None:
        """Request cancellation of the running worker."""
        if self._worker:
            self._worker.cancel()
            self.status_label.setText("Cancelling...")

    def _on_status(self, message: str) -> None:
        self.status_label.setText(message)

    def _on_progress(self, source_path: str, fraction: float) -> None:
        # Overall progress = (completed files + current file fraction) / total files
        overall = (self._completed_files + fraction) / self._total_files
        self.progress_bar.setValue(int(overall * 100))
        basename = Path(source_path).name
        self.status_label.setText(
            f"Analyzing {basename} "
            f"({self._completed_files + 1} of {self._total_files})..."
        )

    def _on_file_complete(self, source_path: str) -> None:
        self._completed_files += 1
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
        self.analysis_complete.emit(moments)

    def _on_error(self, message: str) -> None:
        self.status_label.setText(f"Error: {message}")
        self.progress_bar.hide()
        self.analyze_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self._worker = None

    @property
    def is_analyzing(self) -> bool:
        return self._worker is not None and self._worker.isRunning()
