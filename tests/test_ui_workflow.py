"""E2E UI test: full Import → Analyze → Review → Export workflow."""

from unittest.mock import MagicMock, patch

import pytest

from clipshow.model.moments import DetectedMoment
from clipshow.model.project import VideoSource
from clipshow.ui.main_window import MainWindow
from clipshow.workers.analysis_worker import AnalysisWorker
from clipshow.workers.export_worker import ExportWorker


@pytest.fixture()
def window(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    # Prevent FFmpeg backend from running — it crashes in headless CI
    w.review_panel.video_preview.player.setSource = MagicMock()
    yield w
    w.review_panel.video_preview.cleanup()


def _add_file(window, path="/tmp/test.mp4", duration=10.0):
    source = VideoSource(path=path, duration=duration, width=1920, height=1080)
    window.import_panel.add_source_directly(source)


class TestFullWorkflow:
    """Simulates the entire 4-step UI journey."""

    def test_import_to_analyze_navigation(self, window):
        """Adding files enables Next, clicking Next goes to Analyze tab."""
        assert window.tabs.currentIndex() == 0
        assert window.next_button.isEnabled() is False
        _add_file(window)
        assert window.next_button.isEnabled() is True
        window.next_button.click()
        assert window.tabs.currentIndex() == 1

    @patch.object(AnalysisWorker, "start")
    def test_analyze_creates_worker(self, mock_start, window):
        """Clicking Analyze All creates and starts a worker."""
        _add_file(window)
        window.next_button.click()  # go to Analyze
        window.analyze_panel.analyze_button.click()
        mock_start.assert_called_once()

    def test_analysis_complete_populates_review(self, window):
        """Analysis completion loads segments into review panel."""
        _add_file(window)
        window.next_button.click()  # Analyze tab

        moments = [
            DetectedMoment("/tmp/test.mp4", 1.0, 4.0, 0.8, 0.6, ["scene"]),
            DetectedMoment("/tmp/test.mp4", 6.0, 8.0, 0.5, 0.4, ["audio"]),
        ]
        # Simulate analysis completion
        window._on_analysis_complete(moments)

        assert window.review_panel.segment_list.segment_count == 2

    def test_analysis_complete_populates_export(self, window):
        """Analysis completion also loads segments into export panel."""
        _add_file(window)
        moments = [
            DetectedMoment("/tmp/test.mp4", 1.0, 4.0, 0.8, 0.6, ["scene"]),
        ]
        window._on_analysis_complete(moments)
        assert "1 segments" in window.export_panel.summary_label.text()

    def test_navigate_all_four_tabs(self, window):
        """Can navigate through all four tabs with files loaded."""
        _add_file(window)
        assert window.tabs.currentIndex() == 0
        window.next_button.click()
        assert window.tabs.currentIndex() == 1
        window.next_button.click()
        assert window.tabs.currentIndex() == 2
        window.next_button.click()
        assert window.tabs.currentIndex() == 3
        assert window.next_button.isEnabled() is False

    def test_back_navigation(self, window):
        """Can navigate back through tabs."""
        _add_file(window)
        for _ in range(3):
            window.next_button.click()
        assert window.tabs.currentIndex() == 3
        window.back_button.click()
        assert window.tabs.currentIndex() == 2
        window.back_button.click()
        assert window.tabs.currentIndex() == 1
        window.back_button.click()
        assert window.tabs.currentIndex() == 0
        assert window.back_button.isEnabled() is False

    def test_review_trim_modifies_segment(self, window, qtbot):
        """Trimming a segment modifies its end time."""
        _add_file(window)
        moments = [
            DetectedMoment("/tmp/test.mp4", 1.0, 4.0, 0.8, 0.6, ["scene"]),
            DetectedMoment("/tmp/test.mp4", 6.0, 8.0, 0.5, 0.4, ["audio"]),
        ]
        window._on_analysis_complete(moments)

        # Select a segment and trim directly
        window.review_panel.segment_list.list_widget.setCurrentRow(0)
        window.review_panel._nudge_trim("end", 0.5)

        # Verify the segment was modified
        seg = window.review_panel.segments[0]
        assert seg.end_time == pytest.approx(4.5)

    def test_review_exclude_syncs_to_export(self, window):
        """Excluding a segment in review updates export summary."""
        _add_file(window)
        moments = [
            DetectedMoment("/tmp/test.mp4", 1.0, 4.0, 0.8, 0.6, ["scene"]),
            DetectedMoment("/tmp/test.mp4", 6.0, 8.0, 0.5, 0.4, ["audio"]),
        ]
        window._on_analysis_complete(moments)

        # Uncheck first segment
        item = window.review_panel.segment_list.list_widget.item(0)
        widget = window.review_panel.segment_list.list_widget.itemWidget(item)
        widget.checkbox.setChecked(False)

        assert "1 segments" in window.export_panel.summary_label.text()

    @patch.object(ExportWorker, "start")
    def test_export_creates_worker(self, mock_start, window):
        """Clicking Export launches an ExportWorker."""
        _add_file(window)
        moments = [
            DetectedMoment("/tmp/test.mp4", 1.0, 4.0, 0.8, 0.6, ["scene"]),
        ]
        window._on_analysis_complete(moments)
        window.export_panel.start_export()
        mock_start.assert_called_once()

    def test_export_complete_signal(self, window, qtbot):
        """Export completion emits the signal correctly."""
        with qtbot.waitSignal(window.export_panel.export_complete, timeout=1000):
            window.export_panel._on_complete("/tmp/output.mp4")

    def test_project_shared_across_panels(self, window):
        """Import panel and analyze panel share the same Project."""
        _add_file(window, "/tmp/a.mp4")
        assert len(window.project.sources) == 1
        assert window.project is window.import_panel.project
        assert window.project is window.analyze_panel.project


class TestMenuBar:
    def test_preferences_action_exists(self, window):
        """Edit > Preferences menu action should exist."""
        assert window.preferences_action is not None
        assert "Preferences" in window.preferences_action.text()
