"""Settings dialog: modal editor for all ClipShow settings."""

from __future__ import annotations

from dataclasses import asdict

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from clipshow.config import Settings
from clipshow.ui.prompt_editor import PromptEditor

SLIDER_SCALE = 100

DETECTOR_NAMES = ["scene", "audio", "motion", "semantic", "emotion"]
WEIGHT_FIELDS = {
    "scene": "scene_weight",
    "audio": "audio_weight",
    "motion": "motion_weight",
    "semantic": "semantic_weight",
    "emotion": "emotion_weight",
}

CODEC_OPTIONS = ["libx264", "libx265", "mpeg4"]


class SettingsDialog(QDialog):
    """Modal dialog for editing all ClipShow settings."""

    def __init__(self, settings: Settings, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setMinimumSize(500, 550)
        self.settings = settings
        self._snapshot = asdict(settings)

        self._weight_sliders: dict[str, QSlider] = {}
        self._weight_labels: dict[str, QLabel] = {}

        self._setup_ui()
        self._load_from_settings()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # ── Detector Weights ──────────────────────────────────────
        weights_group = QGroupBox("Detector Weights")
        weights_layout = QFormLayout()
        for name in DETECTOR_NAMES:
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, SLIDER_SCALE)
            label = QLabel()
            row = QHBoxLayout()
            row.addWidget(slider)
            row.addWidget(label)
            weights_layout.addRow(f"{name.capitalize()}:", row)
            self._weight_sliders[name] = slider
            self._weight_labels[name] = label
            slider.valueChanged.connect(
                lambda v, n=name: self._on_weight_slider(n, v)
            )
        weights_group.setLayout(weights_layout)
        layout.addWidget(weights_group)

        # ── Semantic Prompts ──────────────────────────────────────
        prompts_group = QGroupBox("Semantic Prompts")
        prompts_layout = QVBoxLayout()
        self.prompt_editor = PromptEditor(self.settings.semantic_prompts)
        prompts_layout.addWidget(self.prompt_editor)
        prompts_group.setLayout(prompts_layout)
        layout.addWidget(prompts_group)

        # ── Segment Selection ─────────────────────────────────────
        segment_group = QGroupBox("Segment Selection")
        segment_layout = QFormLayout()

        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, SLIDER_SCALE)
        self.threshold_label = QLabel()
        row = QHBoxLayout()
        row.addWidget(self.threshold_slider)
        row.addWidget(self.threshold_label)
        segment_layout.addRow("Score threshold:", row)
        self.threshold_slider.valueChanged.connect(self._on_threshold_slider)

        self.pre_padding_spin = QDoubleSpinBox()
        self.pre_padding_spin.setRange(0.0, 10.0)
        self.pre_padding_spin.setSingleStep(0.5)
        self.pre_padding_spin.setSuffix(" s")
        segment_layout.addRow("Pre-padding:", self.pre_padding_spin)

        self.post_padding_spin = QDoubleSpinBox()
        self.post_padding_spin.setRange(0.0, 10.0)
        self.post_padding_spin.setSingleStep(0.5)
        self.post_padding_spin.setSuffix(" s")
        segment_layout.addRow("Post-padding:", self.post_padding_spin)

        self.min_duration_spin = QDoubleSpinBox()
        self.min_duration_spin.setRange(0.1, 60.0)
        self.min_duration_spin.setSingleStep(0.5)
        self.min_duration_spin.setSuffix(" s")
        segment_layout.addRow("Min duration:", self.min_duration_spin)

        self.max_duration_spin = QDoubleSpinBox()
        self.max_duration_spin.setRange(1.0, 300.0)
        self.max_duration_spin.setSingleStep(1.0)
        self.max_duration_spin.setSuffix(" s")
        segment_layout.addRow("Max duration:", self.max_duration_spin)

        segment_group.setLayout(segment_layout)
        layout.addWidget(segment_group)

        # ── Output Settings ───────────────────────────────────────
        output_group = QGroupBox("Output Settings")
        output_layout = QFormLayout()

        self.codec_combo = QComboBox()
        self.codec_combo.addItems(CODEC_OPTIONS)
        output_layout.addRow("Codec:", self.codec_combo)

        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setRange(1.0, 120.0)
        self.fps_spin.setSingleStep(1.0)
        output_layout.addRow("FPS:", self.fps_spin)

        self.bitrate_edit = QLineEdit()
        self.bitrate_edit.setPlaceholderText("e.g. 8M")
        output_layout.addRow("Bitrate:", self.bitrate_edit)

        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, 64)
        self.workers_spin.setSpecialValueText("Auto (CPU count)")
        output_layout.addRow("Max workers:", self.workers_spin)

        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        # ── Buttons ───────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self.reset_button = QPushButton("Reset to Defaults")
        self.save_button = QPushButton("Save")
        self.cancel_button = QPushButton("Cancel")
        self.save_button.setDefault(True)
        btn_row.addWidget(self.reset_button)
        btn_row.addStretch()
        btn_row.addWidget(self.cancel_button)
        btn_row.addWidget(self.save_button)
        layout.addLayout(btn_row)

        self.save_button.clicked.connect(self._on_save)
        self.cancel_button.clicked.connect(self._on_cancel)
        self.reset_button.clicked.connect(self._on_reset)

    # ── Load / Apply ──────────────────────────────────────────────

    def _load_from_settings(self) -> None:
        """Populate all controls from self.settings."""
        for name in DETECTOR_NAMES:
            field = WEIGHT_FIELDS[name]
            val = getattr(self.settings, field)
            self._weight_sliders[name].setValue(int(val * SLIDER_SCALE))

        self.prompt_editor.prompts = self.settings.semantic_prompts

        self.threshold_slider.setValue(int(self.settings.score_threshold * SLIDER_SCALE))
        self.pre_padding_spin.setValue(self.settings.pre_padding_sec)
        self.post_padding_spin.setValue(self.settings.post_padding_sec)
        self.min_duration_spin.setValue(self.settings.min_segment_duration_sec)
        self.max_duration_spin.setValue(self.settings.max_segment_duration_sec)

        codec = self.settings.output_codec
        idx = CODEC_OPTIONS.index(codec) if codec in CODEC_OPTIONS else 0
        self.codec_combo.setCurrentIndex(idx)
        self.fps_spin.setValue(self.settings.output_fps)
        self.bitrate_edit.setText(self.settings.output_bitrate)
        self.workers_spin.setValue(self.settings.max_workers)

    def _apply_to_settings(self) -> None:
        """Write all control values back to self.settings."""
        for name in DETECTOR_NAMES:
            field = WEIGHT_FIELDS[name]
            setattr(self.settings, field, self._weight_sliders[name].value() / SLIDER_SCALE)

        self.settings.semantic_prompts = self.prompt_editor.prompts

        self.settings.score_threshold = self.threshold_slider.value() / SLIDER_SCALE
        self.settings.pre_padding_sec = self.pre_padding_spin.value()
        self.settings.post_padding_sec = self.post_padding_spin.value()
        self.settings.min_segment_duration_sec = self.min_duration_spin.value()
        self.settings.max_segment_duration_sec = self.max_duration_spin.value()

        self.settings.output_codec = self.codec_combo.currentText()
        self.settings.output_fps = self.fps_spin.value()
        self.settings.output_bitrate = self.bitrate_edit.text().strip() or "8M"
        self.settings.max_workers = self.workers_spin.value()

    # ── Slider helpers ────────────────────────────────────────────

    def _on_weight_slider(self, name: str, value: int) -> None:
        self._weight_labels[name].setText(f"{value / SLIDER_SCALE:.2f}")

    def _on_threshold_slider(self, value: int) -> None:
        self.threshold_label.setText(f"{value / SLIDER_SCALE:.2f}")

    # ── Button actions ────────────────────────────────────────────

    def _on_save(self) -> None:
        self._apply_to_settings()
        self.settings.save()
        self.accept()

    def _on_cancel(self) -> None:
        # Restore snapshot
        for key, val in self._snapshot.items():
            if hasattr(self.settings, key):
                setattr(self.settings, key, val)
        self.reject()

    def _on_reset(self) -> None:
        defaults = Settings()
        for key, val in asdict(defaults).items():
            if hasattr(self.settings, key):
                setattr(self.settings, key, val)
        self._load_from_settings()
