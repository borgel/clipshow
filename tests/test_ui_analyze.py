"""UI tests: analyze panel settings, worker signals, and progress."""

from unittest.mock import MagicMock, patch

import pytest

from clipshow.config import Settings
from clipshow.model.moments import DetectedMoment
from clipshow.model.project import Project, VideoSource
from clipshow.ui.analyze_panel import SLIDER_SCALE, AnalyzePanel
from clipshow.workers.analysis_worker import AnalysisWorker


@pytest.fixture()
def project_with_files():
    project = Project()
    project.add_source(VideoSource(path="/tmp/a.mp4", duration=10.0, width=1920, height=1080))
    project.add_source(VideoSource(path="/tmp/b.mp4", duration=5.0, width=1280, height=720))
    return project


@pytest.fixture()
def panel(qtbot, project_with_files):
    settings = Settings()
    p = AnalyzePanel(project_with_files, settings)
    qtbot.addWidget(p)
    return p


@pytest.fixture()
def empty_panel(qtbot):
    p = AnalyzePanel(Project(), Settings())
    qtbot.addWidget(p)
    return p


class TestInitialState:
    def test_sliders_match_defaults(self, panel):
        s = Settings()
        assert panel.scene_slider.value() == int(s.scene_weight * SLIDER_SCALE)
        assert panel.audio_slider.value() == int(s.audio_weight * SLIDER_SCALE)
        assert panel.motion_slider.value() == int(s.motion_weight * SLIDER_SCALE)

    def test_threshold_matches_default(self, panel):
        s = Settings()
        assert panel.threshold_slider.value() == int(s.score_threshold * SLIDER_SCALE)

    def test_checkboxes_reflect_weights(self, panel):
        assert panel.scene_check.isChecked() is True
        assert panel.audio_check.isChecked() is True
        assert panel.motion_check.isChecked() is True

    def test_analyze_button_enabled(self, panel):
        assert panel.analyze_button.isEnabled() is True

    def test_cancel_button_disabled(self, panel):
        assert panel.cancel_button.isEnabled() is False

    def test_not_analyzing(self, panel):
        assert panel.is_analyzing is False


class TestSliderBinding:
    def test_scene_slider_updates_settings(self, panel):
        panel.scene_slider.setValue(60)
        assert panel.settings.scene_weight == pytest.approx(0.6)

    def test_audio_slider_updates_settings(self, panel):
        panel.audio_slider.setValue(40)
        assert panel.settings.audio_weight == pytest.approx(0.4)

    def test_motion_slider_updates_settings(self, panel):
        panel.motion_slider.setValue(80)
        assert panel.settings.motion_weight == pytest.approx(0.8)

    def test_threshold_slider_updates_settings(self, panel):
        panel.threshold_slider.setValue(70)
        assert panel.settings.score_threshold == pytest.approx(0.7)

    def test_slider_updates_label(self, panel):
        panel.scene_slider.setValue(45)
        assert panel.scene_label.text() == "0.45"

    def test_threshold_updates_label(self, panel):
        panel.threshold_slider.setValue(55)
        assert panel.threshold_label.text() == "0.55"


class TestCheckboxBehavior:
    def test_uncheck_disables_slider(self, panel):
        panel.scene_check.setChecked(False)
        assert panel.scene_slider.isEnabled() is False

    def test_uncheck_zeros_weight(self, panel):
        panel.scene_check.setChecked(False)
        assert panel.settings.scene_weight == 0.0

    def test_recheck_enables_slider(self, panel):
        panel.scene_check.setChecked(False)
        panel.scene_check.setChecked(True)
        assert panel.scene_slider.isEnabled() is True


class TestStartAnalysis:
    def test_no_sources_does_nothing(self, empty_panel):
        """Start analysis with no files should not crash or create a worker."""
        empty_panel.start_analysis()
        assert empty_panel._worker is None

    @patch.object(AnalysisWorker, "start")
    def test_creates_progress_bars(self, mock_start, panel):
        panel.start_analysis()
        assert len(panel._progress_bars) == 2
        assert "/tmp/a.mp4" in panel._progress_bars
        assert "/tmp/b.mp4" in panel._progress_bars

    @patch.object(AnalysisWorker, "start")
    def test_disables_analyze_button(self, mock_start, panel):
        panel.start_analysis()
        assert panel.analyze_button.isEnabled() is False

    @patch.object(AnalysisWorker, "start")
    def test_enables_cancel_button(self, mock_start, panel):
        panel.start_analysis()
        assert panel.cancel_button.isEnabled() is True


class TestProgressSignals:
    def test_progress_updates_bar(self, panel):
        panel._progress_bars["/tmp/a.mp4"] = MagicMock()
        panel._on_progress("/tmp/a.mp4", 0.5)
        panel._progress_bars["/tmp/a.mp4"].setValue.assert_called_with(50)

    def test_file_complete_sets_100(self, panel):
        panel._progress_bars["/tmp/a.mp4"] = MagicMock()
        panel._on_file_complete("/tmp/a.mp4")
        panel._progress_bars["/tmp/a.mp4"].setValue.assert_called_with(100)

    def test_unknown_path_no_crash(self, panel):
        """Progress for unknown path should not crash."""
        panel._on_progress("/tmp/unknown.mp4", 0.5)
        panel._on_file_complete("/tmp/unknown.mp4")


class TestAnalysisComplete:
    def test_emits_signal(self, panel, qtbot):
        moments = [
            DetectedMoment("/tmp/a.mp4", 1.0, 3.0, 0.8, 0.6, ["scene"]),
        ]
        with qtbot.waitSignal(panel.analysis_complete, timeout=1000):
            panel._on_all_complete(moments)

    def test_re_enables_analyze(self, panel):
        panel.analyze_button.setEnabled(False)
        panel._on_all_complete([])
        assert panel.analyze_button.isEnabled() is True

    def test_disables_cancel(self, panel):
        panel.cancel_button.setEnabled(True)
        panel._on_all_complete([])
        assert panel.cancel_button.isEnabled() is False

    def test_clears_worker(self, panel):
        panel._worker = MagicMock()
        panel._on_all_complete([])
        assert panel._worker is None


class TestErrorHandling:
    def test_error_re_enables_analyze(self, panel):
        panel.analyze_button.setEnabled(False)
        panel._on_error("something broke")
        assert panel.analyze_button.isEnabled() is True

    def test_error_disables_cancel(self, panel):
        panel.cancel_button.setEnabled(True)
        panel._on_error("something broke")
        assert panel.cancel_button.isEnabled() is False

    def test_error_clears_worker(self, panel):
        panel._worker = MagicMock()
        panel._on_error("something broke")
        assert panel._worker is None


class TestCancelAnalysis:
    @patch.object(AnalysisWorker, "start")
    def test_cancel_calls_worker_cancel(self, mock_start, panel):
        panel.start_analysis()
        panel._worker.cancel = MagicMock()
        panel.cancel_analysis()
        panel._worker.cancel.assert_called_once()

    def test_cancel_without_worker_no_crash(self, panel):
        panel.cancel_analysis()  # should not raise
