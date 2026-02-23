"""UI tests: import panel file management and button states."""

import pytest

from clipshow.model.project import Project, VideoSource
from clipshow.ui.import_panel import ImportPanel


@pytest.fixture()
def panel(qtbot):
    """Fresh ImportPanel with an empty project."""
    project = Project()
    p = ImportPanel(project)
    qtbot.addWidget(p)
    return p


def _make_source(path="/tmp/vid1.mp4", duration=10.0, w=1920, h=1080):
    return VideoSource(path=path, duration=duration, width=w, height=h)


class TestInitialState:
    def test_table_empty(self, panel):
        assert panel.file_table.rowCount() == 0

    def test_file_count_zero(self, panel):
        assert panel.file_count == 0

    def test_remove_disabled(self, panel):
        assert panel.remove_button.isEnabled() is False

    def test_clear_disabled(self, panel):
        assert panel.clear_button.isEnabled() is False

    def test_browse_enabled(self, panel):
        assert panel.browse_button.isEnabled() is True


class TestAddFiles:
    def test_add_source_directly(self, panel):
        panel.add_source_directly(_make_source())
        assert panel.file_count == 1
        assert panel.file_table.item(0, 0).text() == "vid1.mp4"

    def test_add_populates_duration(self, panel):
        panel.add_source_directly(_make_source(duration=125.0))
        assert panel.file_table.item(0, 1).text() == "2:05"

    def test_add_populates_resolution(self, panel):
        panel.add_source_directly(_make_source(w=3840, h=2160))
        assert panel.file_table.item(0, 2).text() == "3840x2160"

    def test_add_populates_path(self, panel):
        panel.add_source_directly(_make_source(path="/movies/clip.mp4"))
        assert panel.file_table.item(0, 3).text() == "/movies/clip.mp4"

    def test_add_multiple(self, panel):
        panel.add_source_directly(_make_source(path="/tmp/a.mp4"))
        panel.add_source_directly(_make_source(path="/tmp/b.mp4"))
        panel.add_source_directly(_make_source(path="/tmp/c.mp4"))
        assert panel.file_count == 3

    def test_duplicate_ignored(self, panel):
        panel.add_source_directly(_make_source(path="/tmp/a.mp4"))
        panel.add_source_directly(_make_source(path="/tmp/a.mp4"))
        assert panel.file_count == 1

    def test_add_updates_project(self, panel):
        panel.add_source_directly(_make_source(path="/tmp/x.mp4"))
        assert len(panel.project.sources) == 1
        assert panel.project.sources[0].path == "/tmp/x.mp4"

    def test_clear_enabled_after_add(self, panel):
        panel.add_source_directly(_make_source())
        assert panel.clear_button.isEnabled() is True

    def test_zero_duration_shows_placeholder(self, panel):
        panel.add_source_directly(_make_source(duration=0.0))
        assert panel.file_table.item(0, 1).text() == "--:--"

    def test_files_changed_signal(self, panel, qtbot):
        with qtbot.waitSignal(panel.files_changed, timeout=1000):
            panel.add_source_directly(_make_source())


class TestRemoveFiles:
    def test_remove_selected(self, panel, qtbot):
        panel.add_source_directly(_make_source(path="/tmp/a.mp4"))
        panel.add_source_directly(_make_source(path="/tmp/b.mp4"))
        # Select first row
        panel.file_table.selectRow(0)
        assert panel.remove_button.isEnabled() is True
        panel.remove_button.click()
        assert panel.file_count == 1
        assert panel.file_table.item(0, 0).text() == "b.mp4"

    def test_remove_updates_project(self, panel):
        panel.add_source_directly(_make_source(path="/tmp/a.mp4"))
        panel.add_source_directly(_make_source(path="/tmp/b.mp4"))
        panel.file_table.selectRow(0)
        panel.remove_button.click()
        assert len(panel.project.sources) == 1
        assert panel.project.sources[0].path == "/tmp/b.mp4"

    def test_remove_disabled_after_removing_all_selected(self, panel):
        panel.add_source_directly(_make_source(path="/tmp/a.mp4"))
        panel.file_table.selectRow(0)
        panel.remove_button.click()
        assert panel.remove_button.isEnabled() is False

    def test_remove_emits_files_changed(self, panel, qtbot):
        panel.add_source_directly(_make_source())
        panel.file_table.selectRow(0)
        with qtbot.waitSignal(panel.files_changed, timeout=1000):
            panel.remove_button.click()


class TestClearAll:
    def test_clear_removes_all(self, panel):
        panel.add_source_directly(_make_source(path="/tmp/a.mp4"))
        panel.add_source_directly(_make_source(path="/tmp/b.mp4"))
        panel.clear_button.click()
        assert panel.file_count == 0

    def test_clear_updates_project(self, panel):
        panel.add_source_directly(_make_source(path="/tmp/a.mp4"))
        panel.clear_button.click()
        assert len(panel.project.sources) == 0

    def test_clear_disables_buttons(self, panel):
        panel.add_source_directly(_make_source())
        panel.clear_button.click()
        assert panel.clear_button.isEnabled() is False
        assert panel.remove_button.isEnabled() is False

    def test_clear_emits_files_changed(self, panel, qtbot):
        panel.add_source_directly(_make_source())
        with qtbot.waitSignal(panel.files_changed, timeout=1000):
            panel.clear_button.click()


class TestSelectionStates:
    def test_remove_disabled_without_selection(self, panel):
        panel.add_source_directly(_make_source())
        assert panel.remove_button.isEnabled() is False

    def test_remove_enabled_with_selection(self, panel):
        panel.add_source_directly(_make_source())
        panel.file_table.selectRow(0)
        assert panel.remove_button.isEnabled() is True

    def test_remove_disabled_after_deselect(self, panel):
        panel.add_source_directly(_make_source())
        panel.file_table.selectRow(0)
        panel.file_table.clearSelection()
        assert panel.remove_button.isEnabled() is False


class TestDurationFormat:
    def test_short_duration(self, panel):
        panel.add_source_directly(_make_source(duration=5.0))
        assert panel.file_table.item(0, 1).text() == "0:05"

    def test_one_minute(self, panel):
        panel.add_source_directly(_make_source(duration=60.0))
        assert panel.file_table.item(0, 1).text() == "1:00"

    def test_long_duration(self, panel):
        panel.add_source_directly(_make_source(duration=3661.0))
        assert panel.file_table.item(0, 1).text() == "61:01"
