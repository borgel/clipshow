"""UI tests: main window layout and tab navigation."""

from clipshow.model.moments import DetectedMoment
from clipshow.model.project import VideoSource
from clipshow.ui.main_window import MainWindow


def _window_with_file(qtbot):
    """Create a MainWindow with one file loaded so Next is enabled."""
    window = MainWindow()
    qtbot.addWidget(window)
    source = VideoSource(path="/tmp/test.mp4", duration=10.0, width=1920, height=1080)
    window.import_panel.add_source_directly(source)
    return window


class TestMainWindow:
    def test_window_creates(self, qtbot):
        window = MainWindow()
        qtbot.addWidget(window)
        assert window.windowTitle() == "ClipShow"

    def test_has_four_tabs(self, qtbot):
        window = MainWindow()
        qtbot.addWidget(window)
        assert window.tabs.count() == 4
        assert window.tabs.tabText(0) == "1. Import"
        assert window.tabs.tabText(1) == "2. Analyze"
        assert window.tabs.tabText(2) == "3. Review"
        assert window.tabs.tabText(3) == "4. Export"

    def test_only_import_tab_enabled_initially(self, qtbot):
        window = MainWindow()
        qtbot.addWidget(window)
        assert window.tabs.isTabEnabled(0) is True
        assert window.tabs.isTabEnabled(1) is False
        assert window.tabs.isTabEnabled(2) is False
        assert window.tabs.isTabEnabled(3) is False

    def test_back_disabled_on_first_tab(self, qtbot):
        window = MainWindow()
        qtbot.addWidget(window)
        assert window.back_button.isEnabled() is False

    def test_next_disabled_with_no_files(self, qtbot):
        window = MainWindow()
        qtbot.addWidget(window)
        assert window.next_button.isEnabled() is False

    def test_next_enabled_after_file_added(self, qtbot):
        window = _window_with_file(qtbot)
        assert window.next_button.isEnabled() is True

    def test_next_advances_tab(self, qtbot):
        window = _window_with_file(qtbot)
        assert window.tabs.currentIndex() == 0
        window.next_button.click()
        assert window.tabs.currentIndex() == 1
        assert window.tabs.isTabEnabled(1) is True

    def test_back_returns_to_previous(self, qtbot):
        window = _window_with_file(qtbot)
        window.next_button.click()
        assert window.tabs.currentIndex() == 1
        window.back_button.click()
        assert window.tabs.currentIndex() == 0

    def test_next_disabled_on_last_tab(self, qtbot):
        window = _window_with_file(qtbot)
        window.next_button.click()  # -> Analyze (1)
        # Simulate analysis completing so Next works
        moments = [DetectedMoment("/tmp/test.mp4", 1.0, 3.0, 0.8, 0.6, ["scene"])]
        window.analyze_panel._on_all_complete(moments)
        window.next_button.click()  # -> Review (2)
        window.next_button.click()  # -> Export (3)
        assert window.tabs.currentIndex() == 3
        assert window.next_button.isEnabled() is False

    def test_sequential_tab_enabling(self, qtbot):
        """Tabs should enable sequentially as Next is clicked."""
        window = _window_with_file(qtbot)
        window.next_button.click()
        assert window.tabs.currentIndex() == 1
        assert window.tabs.isTabEnabled(1) is True
        # Simulate analysis completing so Next works on Analyze tab
        moments = [DetectedMoment("/tmp/test.mp4", 1.0, 3.0, 0.8, 0.6, ["scene"])]
        window.analyze_panel._on_all_complete(moments)
        window.next_button.click()
        assert window.tabs.currentIndex() == 2
        assert window.tabs.isTabEnabled(2) is True
        window.next_button.click()
        assert window.tabs.currentIndex() == 3
        assert window.tabs.isTabEnabled(3) is True

    def test_minimum_size(self, qtbot):
        window = MainWindow()
        qtbot.addWidget(window)
        assert window.minimumWidth() == 800
        assert window.minimumHeight() == 600
