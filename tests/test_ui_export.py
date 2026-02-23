"""UI tests: export panel settings, worker flow, and progress."""

from unittest.mock import MagicMock, patch

import pytest

from clipshow.model.moments import HighlightSegment
from clipshow.ui.export_panel import ExportPanel
from clipshow.workers.export_worker import ExportWorker


def _make_segments():
    return [
        HighlightSegment("/tmp/a.mp4", 1.0, 4.0, 0.8, included=True, order=0),
        HighlightSegment("/tmp/b.mp4", 2.0, 5.0, 0.6, included=True, order=1),
    ]


@pytest.fixture()
def panel(qtbot):
    p = ExportPanel()
    qtbot.addWidget(p)
    return p


@pytest.fixture()
def loaded_panel(qtbot):
    p = ExportPanel()
    qtbot.addWidget(p)
    p.set_segments(_make_segments())
    return p


class TestInitialState:
    def test_export_disabled_no_segments(self, panel):
        assert panel.export_button.isEnabled() is False

    def test_cancel_disabled(self, panel):
        assert panel.cancel_button.isEnabled() is False

    def test_progress_hidden(self, panel):
        assert panel.progress_bar.isHidden() is True

    def test_summary_no_segments(self, panel):
        assert "No segments" in panel.summary_label.text()

    def test_default_path(self, panel):
        assert panel.path_edit.text() == "highlight_reel.mp4"

    def test_default_fps(self, panel):
        assert panel.fps_spin.value() == 30

    def test_default_bitrate(self, panel):
        assert panel.bitrate_edit.text() == "8M"

    def test_not_exporting(self, panel):
        assert panel.is_exporting is False


class TestSettingsBinding:
    def test_fps_updates_settings(self, panel):
        panel.fps_spin.setValue(60)
        assert panel.export_settings.fps == 60.0

    def test_bitrate_updates_settings(self, panel):
        panel.bitrate_edit.setText("12M")
        assert panel.export_settings.bitrate == "12M"

    def test_path_updates_settings(self, panel):
        panel.path_edit.setText("/out/reel.mp4")
        assert panel.export_settings.output_path == "/out/reel.mp4"


class TestSetSegments:
    def test_summary_shows_count(self, loaded_panel):
        assert "2 segments" in loaded_panel.summary_label.text()

    def test_summary_shows_duration(self, loaded_panel):
        assert "6.0s" in loaded_panel.summary_label.text()

    def test_export_enabled(self, loaded_panel):
        assert loaded_panel.export_button.isEnabled() is True

    def test_excluded_not_counted(self, panel):
        segs = _make_segments()
        segs[0].included = False
        panel.set_segments(segs)
        assert "1 segments" in panel.summary_label.text()

    def test_all_excluded_disables_export(self, panel):
        segs = _make_segments()
        for s in segs:
            s.included = False
        panel.set_segments(segs)
        assert panel.export_button.isEnabled() is False


class TestStartExport:
    @patch.object(ExportWorker, "start")
    def test_shows_progress_bar(self, mock_start, loaded_panel):
        loaded_panel.start_export()
        assert loaded_panel.progress_bar.isHidden() is False

    @patch.object(ExportWorker, "start")
    def test_disables_export_button(self, mock_start, loaded_panel):
        loaded_panel.start_export()
        assert loaded_panel.export_button.isEnabled() is False

    @patch.object(ExportWorker, "start")
    def test_enables_cancel_button(self, mock_start, loaded_panel):
        loaded_panel.start_export()
        assert loaded_panel.cancel_button.isEnabled() is True

    def test_no_segments_does_nothing(self, panel):
        panel.start_export()
        assert panel._worker is None


class TestProgressSignals:
    def test_progress_updates_bar(self, loaded_panel):
        loaded_panel.progress_bar.setVisible(True)
        loaded_panel._on_progress(0.5)
        assert loaded_panel.progress_bar.value() == 50

    def test_complete_sets_100(self, loaded_panel):
        loaded_panel.progress_bar.setVisible(True)
        loaded_panel._on_complete("/out/reel.mp4")
        assert loaded_panel.progress_bar.value() == 100

    def test_complete_emits_signal(self, loaded_panel, qtbot):
        with qtbot.waitSignal(loaded_panel.export_complete, timeout=1000):
            loaded_panel._on_complete("/out/reel.mp4")

    def test_complete_re_enables_export(self, loaded_panel):
        loaded_panel.export_button.setEnabled(False)
        loaded_panel._on_complete("/out/reel.mp4")
        assert loaded_panel.export_button.isEnabled() is True

    def test_complete_clears_worker(self, loaded_panel):
        loaded_panel._worker = MagicMock()
        loaded_panel._on_complete("/out/reel.mp4")
        assert loaded_panel._worker is None


class TestErrorHandling:
    def test_error_re_enables_export(self, loaded_panel):
        loaded_panel.export_button.setEnabled(False)
        loaded_panel._on_error("ffmpeg died")
        assert loaded_panel.export_button.isEnabled() is True

    def test_error_hides_progress(self, loaded_panel):
        loaded_panel.progress_bar.setVisible(True)
        loaded_panel._on_error("ffmpeg died")
        assert loaded_panel.progress_bar.isHidden() is True

    def test_error_emits_signal(self, loaded_panel, qtbot):
        with qtbot.waitSignal(loaded_panel.export_error, timeout=1000):
            loaded_panel._on_error("ffmpeg died")

    def test_error_clears_worker(self, loaded_panel):
        loaded_panel._worker = MagicMock()
        loaded_panel._on_error("ffmpeg died")
        assert loaded_panel._worker is None
