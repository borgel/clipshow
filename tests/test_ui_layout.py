"""UI tests: main window layout and tab navigation."""

from clipshow.ui.main_window import MainWindow


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

    def test_next_enabled_on_first_tab(self, qtbot):
        window = MainWindow()
        qtbot.addWidget(window)
        assert window.next_button.isEnabled() is True

    def test_next_advances_tab(self, qtbot):
        window = MainWindow()
        qtbot.addWidget(window)
        assert window.tabs.currentIndex() == 0
        window.next_button.click()
        assert window.tabs.currentIndex() == 1
        assert window.tabs.isTabEnabled(1) is True

    def test_back_returns_to_previous(self, qtbot):
        window = MainWindow()
        qtbot.addWidget(window)
        window.next_button.click()
        assert window.tabs.currentIndex() == 1
        window.back_button.click()
        assert window.tabs.currentIndex() == 0

    def test_next_disabled_on_last_tab(self, qtbot):
        window = MainWindow()
        qtbot.addWidget(window)
        for _ in range(3):
            window.next_button.click()
        assert window.tabs.currentIndex() == 3
        assert window.next_button.isEnabled() is False

    def test_sequential_tab_enabling(self, qtbot):
        """Tabs should enable sequentially as Next is clicked."""
        window = MainWindow()
        qtbot.addWidget(window)
        for expected_tab in range(1, 4):
            window.next_button.click()
            assert window.tabs.currentIndex() == expected_tab
            assert window.tabs.isTabEnabled(expected_tab) is True

    def test_minimum_size(self, qtbot):
        window = MainWindow()
        qtbot.addWidget(window)
        assert window.minimumWidth() == 800
        assert window.minimumHeight() == 600
