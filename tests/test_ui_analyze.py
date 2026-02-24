"""UI tests: analyze panel settings, worker signals, and progress."""

import time
from unittest.mock import MagicMock, patch

import pytest

from clipshow.config import Settings
from clipshow.model.moments import DetectedMoment, HighlightSegment
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
        assert panel.semantic_slider.value() == int(s.semantic_weight * SLIDER_SCALE)
        assert panel.emotion_slider.value() == int(s.emotion_weight * SLIDER_SCALE)

    def test_threshold_matches_default(self, panel):
        s = Settings()
        assert panel.threshold_slider.value() == int(s.score_threshold * SLIDER_SCALE)

    def test_checkboxes_reflect_weights(self, panel):
        assert panel.scene_check.isChecked() is True
        assert panel.audio_check.isChecked() is True
        assert panel.motion_check.isChecked() is True
        # semantic defaults to 0.0, so unchecked
        assert panel.semantic_check.isChecked() is False
        # emotion defaults to 0.2, so checked
        assert panel.emotion_check.isChecked() is True

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

    def test_semantic_slider_updates_settings(self, panel):
        panel.semantic_check.setChecked(True)
        panel.semantic_slider.setValue(50)
        assert panel.settings.semantic_weight == pytest.approx(0.5)

    def test_emotion_slider_updates_settings(self, panel):
        panel.emotion_slider.setValue(35)
        assert panel.settings.emotion_weight == pytest.approx(0.35)

    def test_threshold_slider_updates_settings(self, panel):
        panel.threshold_slider.setValue(70)
        assert panel.settings.score_threshold == pytest.approx(0.7)

    def test_slider_updates_label(self, panel):
        panel.scene_slider.setValue(45)
        assert panel.scene_label.text() == "45%"

    def test_threshold_updates_label(self, panel):
        panel.threshold_slider.setValue(55)
        assert panel.threshold_label.text() == "55%"


class TestEditPromptsButton:
    def test_edit_prompts_button_exists(self, panel):
        assert panel.edit_prompts_button is not None
        assert panel.edit_prompts_button.text() == "Edit Prompts\u2026"


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
    def test_shows_status_and_progress(self, mock_start, panel):
        panel.start_analysis()
        assert not panel.progress_bar.isHidden()
        assert "2 videos" in panel.status_label.text()
        assert panel._total_files == 2

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
        panel._total_files = 2
        panel._completed_files = 0
        panel._on_progress("/tmp/a.mp4", 0.5)
        # Overall: (0 + 0.5) / 2 = 25%
        assert panel.progress_bar.value() == 25
        assert "a.mp4" in panel.status_label.text()

    def test_file_complete_updates_status(self, panel):
        panel._total_files = 2
        panel._completed_files = 0
        panel._on_file_complete("/tmp/a.mp4")
        assert panel._completed_files == 1
        assert "Completed a.mp4" in panel.status_label.text()

    def test_progress_with_completed_files(self, panel):
        panel._total_files = 2
        panel._completed_files = 1
        # Simulate first file already complete in the progress dict
        panel._file_progress["/tmp/a.mp4"] = 1.0
        panel._on_progress("/tmp/b.mp4", 0.5)
        # Overall: (1.0 + 0.5) / 2 = 75%
        assert panel.progress_bar.value() == 75


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


class TestHasResults:
    def test_initially_false(self, panel):
        assert panel.has_results is False

    def test_true_after_all_complete(self, panel):
        panel._on_all_complete([])
        assert panel.has_results is True

    def test_still_true_after_error(self, panel):
        """Error doesn't reset has_results — previous results may still be valid."""
        panel._on_all_complete([])
        panel._on_error("something broke")
        assert panel.has_results is True


class TestAnalysisStartedSignal:
    @patch.object(AnalysisWorker, "start")
    def test_emits_on_start(self, mock_start, panel, qtbot):
        with qtbot.waitSignal(panel.analysis_started, timeout=1000):
            panel.start_analysis()

    def test_not_emitted_without_sources(self, empty_panel, qtbot):
        """Starting analysis with no sources should not emit analysis_started."""
        signals = []
        empty_panel.analysis_started.connect(lambda: signals.append(True))
        empty_panel.start_analysis()
        assert signals == []


class TestFileList:
    @patch.object(AnalysisWorker, "start")
    def test_file_list_populated_on_start(self, mock_start, panel):
        panel.start_analysis()
        assert panel.file_list.count() == 2

    @patch.object(AnalysisWorker, "start")
    def test_first_progress_marks_analyzing(self, mock_start, panel):
        """File is marked analyzing on first progress signal."""
        panel.start_analysis()
        panel._on_progress("/tmp/a.mp4", 0.1)
        text = panel.file_list.item(0).text()
        assert "a.mp4" in text
        assert "\u25B6" in text  # play triangle

    @patch.object(AnalysisWorker, "start")
    def test_second_file_pending(self, mock_start, panel):
        panel.start_analysis()
        text = panel.file_list.item(1).text()
        assert "b.mp4" in text
        assert "\u2500" in text  # dash (pending)

    @patch.object(AnalysisWorker, "start")
    def test_file_complete_marks_checkmark(self, mock_start, panel):
        panel.start_analysis()
        panel._on_file_complete("/tmp/a.mp4")
        text = panel.file_list.item(0).text()
        assert "\u2714" in text  # checkmark

    @patch.object(AnalysisWorker, "start")
    def test_parallel_both_analyzing(self, mock_start, panel):
        """Both files can be marked analyzing concurrently."""
        panel.start_analysis()
        panel._on_progress("/tmp/a.mp4", 0.1)
        panel._on_progress("/tmp/b.mp4", 0.1)
        assert "\u25B6" in panel.file_list.item(0).text()
        assert "\u25B6" in panel.file_list.item(1).text()


class TestETAFormatting:
    def test_short_eta(self):
        assert AnalyzePanel._format_eta(45) == "~45s remaining"

    def test_minutes_eta(self):
        assert AnalyzePanel._format_eta(132) == "~2m 12s remaining"

    def test_zero_eta(self):
        assert AnalyzePanel._format_eta(0) == "~0s remaining"


class TestRateAndETA:
    def test_rate_shown_in_status(self, panel):
        """After sufficient elapsed time, status should include rate and ETA."""
        panel._total_files = 1
        panel._completed_files = 0
        panel._total_video_duration = 100.0
        # Simulate 2 seconds elapsed
        panel._analysis_start_time = time.monotonic() - 2.0
        panel._on_progress("/tmp/a.mp4", 0.5)
        text = panel.status_label.text()
        assert "realtime" in text
        assert "remaining" in text

    def test_rate_not_shown_initially(self, panel):
        """When elapsed time is tiny, no rate/ETA shown."""
        panel._total_files = 1
        panel._completed_files = 0
        panel._total_video_duration = 100.0
        panel._analysis_start_time = time.monotonic()
        panel._on_progress("/tmp/a.mp4", 0.001)
        text = panel.status_label.text()
        assert "realtime" not in text


class TestHighlightSegmentDetectors:
    def test_from_moment_copies_detectors(self):
        m = DetectedMoment("/tmp/a.mp4", 1.0, 3.0, 0.8, 0.6, ["scene", "audio"])
        seg = HighlightSegment.from_moment(m, order=0)
        assert seg.detectors == ["scene", "audio"]

    def test_from_moment_default_empty(self):
        m = DetectedMoment("/tmp/a.mp4", 1.0, 3.0, 0.8, 0.6)
        seg = HighlightSegment.from_moment(m, order=0)
        assert seg.detectors == []

    def test_detectors_independent_copy(self):
        """Modifying segment detectors shouldn't affect the moment."""
        m = DetectedMoment("/tmp/a.mp4", 1.0, 3.0, 0.8, 0.6, ["scene"])
        seg = HighlightSegment.from_moment(m, order=0)
        seg.detectors.append("audio")
        assert m.contributing_detectors == ["scene"]
