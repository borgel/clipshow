"""Tests for the Settings Dialog."""

import pytest

from clipshow.config import Settings
from clipshow.ui.settings_dialog import SLIDER_SCALE, SettingsDialog


@pytest.fixture()
def settings():
    return Settings()


@pytest.fixture()
def dialog(qtbot, settings):
    dlg = SettingsDialog(settings)
    qtbot.addWidget(dlg)
    return dlg


class TestInitialValues:
    def test_weight_sliders_match_settings(self, dialog, settings):
        assert dialog._weight_sliders["scene"].value() == int(settings.scene_weight * SLIDER_SCALE)
        assert dialog._weight_sliders["audio"].value() == int(settings.audio_weight * SLIDER_SCALE)
        assert dialog._weight_sliders["motion"].value() == int(
            settings.motion_weight * SLIDER_SCALE
        )
        assert dialog._weight_sliders["semantic"].value() == int(
            settings.semantic_weight * SLIDER_SCALE
        )
        assert dialog._weight_sliders["emotion"].value() == int(
            settings.emotion_weight * SLIDER_SCALE
        )

    def test_threshold_slider_matches(self, dialog, settings):
        assert dialog.threshold_slider.value() == int(settings.score_threshold * SLIDER_SCALE)

    def test_padding_spinboxes_match(self, dialog, settings):
        assert dialog.pre_padding_spin.value() == pytest.approx(settings.pre_padding_sec)
        assert dialog.post_padding_spin.value() == pytest.approx(settings.post_padding_sec)

    def test_duration_spinboxes_match(self, dialog, settings):
        assert dialog.min_duration_spin.value() == pytest.approx(settings.min_segment_duration_sec)
        assert dialog.max_duration_spin.value() == pytest.approx(settings.max_segment_duration_sec)

    def test_codec_combo_matches(self, dialog, settings):
        assert dialog.codec_combo.currentText() == settings.output_codec

    def test_fps_spin_matches(self, dialog, settings):
        assert dialog.fps_spin.value() == pytest.approx(settings.output_fps)

    def test_bitrate_matches(self, dialog, settings):
        assert dialog.bitrate_edit.text() == settings.output_bitrate

    def test_prompt_editor_matches(self, dialog, settings):
        assert dialog.prompt_editor.prompts == settings.semantic_prompts

    def test_negative_prompt_editor_matches(self, dialog, settings):
        assert dialog.negative_prompt_editor.prompts == settings.semantic_negative_prompts


class TestSliderChanges:
    def test_weight_slider_updates_label(self, dialog):
        dialog._weight_sliders["scene"].setValue(75)
        assert dialog._weight_labels["scene"].text() == "0.75"

    def test_threshold_slider_updates_label(self, dialog):
        dialog.threshold_slider.setValue(42)
        assert dialog.threshold_label.text() == "0.42"


class TestSave:
    def test_save_applies_weight_changes(self, dialog, settings, tmp_path):
        dialog._weight_sliders["scene"].setValue(90)
        dialog._weight_sliders["audio"].setValue(10)
        # Redirect save to temp file so we don't pollute real config
        settings.save = lambda path=None: None
        dialog._on_save()
        assert settings.scene_weight == pytest.approx(0.9)
        assert settings.audio_weight == pytest.approx(0.1)

    def test_save_applies_segment_settings(self, dialog, settings):
        dialog.threshold_slider.setValue(65)
        dialog.pre_padding_spin.setValue(2.5)
        dialog.post_padding_spin.setValue(3.0)
        dialog.min_duration_spin.setValue(2.0)
        dialog.max_duration_spin.setValue(30.0)
        settings.save = lambda path=None: None
        dialog._on_save()
        assert settings.score_threshold == pytest.approx(0.65)
        assert settings.pre_padding_sec == pytest.approx(2.5)
        assert settings.post_padding_sec == pytest.approx(3.0)
        assert settings.min_segment_duration_sec == pytest.approx(2.0)
        assert settings.max_segment_duration_sec == pytest.approx(30.0)

    def test_save_applies_output_settings(self, dialog, settings):
        dialog.codec_combo.setCurrentText("libx265")
        dialog.fps_spin.setValue(60.0)
        dialog.bitrate_edit.setText("12M")
        settings.save = lambda path=None: None
        dialog._on_save()
        assert settings.output_codec == "libx265"
        assert settings.output_fps == pytest.approx(60.0)
        assert settings.output_bitrate == "12M"

    def test_save_applies_prompts(self, dialog, settings):
        dialog.prompt_editor.line_edit.setText("custom prompt")
        dialog.prompt_editor.add_button.click()
        settings.save = lambda path=None: None
        dialog._on_save()
        assert "custom prompt" in settings.semantic_prompts

    def test_save_applies_negative_prompts(self, dialog, settings):
        dialog.negative_prompt_editor.line_edit.setText("custom negative")
        dialog.negative_prompt_editor.add_button.click()
        settings.save = lambda path=None: None
        dialog._on_save()
        assert "custom negative" in settings.semantic_negative_prompts


class TestCancel:
    def test_cancel_reverts_changes(self, dialog, settings):
        original_scene = settings.scene_weight
        dialog._weight_sliders["scene"].setValue(99)
        dialog._on_cancel()
        assert settings.scene_weight == pytest.approx(original_scene)

    def test_cancel_reverts_all_fields(self, dialog, settings):
        original_threshold = settings.score_threshold
        original_codec = settings.output_codec
        dialog.threshold_slider.setValue(10)
        dialog.codec_combo.setCurrentText("mpeg4")
        dialog._on_cancel()
        assert settings.score_threshold == pytest.approx(original_threshold)
        assert settings.output_codec == original_codec


class TestResetDefaults:
    def test_reset_restores_defaults(self, dialog, settings):
        dialog._weight_sliders["scene"].setValue(99)
        dialog.threshold_slider.setValue(10)
        dialog._on_reset()
        defaults = Settings()
        assert dialog._weight_sliders["scene"].value() == int(
            defaults.scene_weight * SLIDER_SCALE
        )
        assert dialog.threshold_slider.value() == int(defaults.score_threshold * SLIDER_SCALE)

    def test_reset_restores_output_defaults(self, dialog, settings):
        dialog.codec_combo.setCurrentText("libx265")
        dialog.fps_spin.setValue(120.0)
        dialog._on_reset()
        defaults = Settings()
        assert dialog.codec_combo.currentText() == defaults.output_codec
        assert dialog.fps_spin.value() == pytest.approx(defaults.output_fps)
