"""Main window with 4-step tabbed workflow."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from clipshow.config import Settings
from clipshow.model.moments import HighlightSegment
from clipshow.model.project import Project
from clipshow.ui.analyze_panel import AnalyzePanel
from clipshow.ui.export_panel import ExportPanel
from clipshow.ui.import_panel import ImportPanel
from clipshow.ui.review_panel import ReviewPanel
from clipshow.ui.settings_dialog import SettingsDialog


class MainWindow(QMainWindow):
    """ClipShow main window with Import → Analyze → Review → Export tabs."""

    def __init__(
        self,
        project: Project | None = None,
        settings: Settings | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("ClipShow")
        self.setMinimumSize(800, 600)

        self.project = project or Project()
        self.settings = settings or Settings()

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Tab widget
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Create panels
        self.import_panel = ImportPanel(self.project)
        self.analyze_panel = AnalyzePanel(self.project, self.settings)
        self.review_panel = ReviewPanel()
        self.export_panel = ExportPanel(self.project.export_settings)

        self.tabs.addTab(self.import_panel, "1. Import")
        self.tabs.addTab(self.analyze_panel, "2. Analyze")
        self.tabs.addTab(self.review_panel, "3. Review")
        self.tabs.addTab(self.export_panel, "4. Export")

        # Disable all tabs except Import initially
        for i in range(1, self.tabs.count()):
            self.tabs.setTabEnabled(i, False)

        # Navigation buttons
        nav_layout = QHBoxLayout()
        self.back_button = QPushButton("Back")
        self.next_button = QPushButton("Next")

        self.back_button.setEnabled(False)
        self.next_button.setEnabled(False)  # disabled until files imported
        nav_layout.addStretch()
        nav_layout.addWidget(self.back_button)
        nav_layout.addWidget(self.next_button)
        layout.addLayout(nav_layout)

        # Menu bar
        menu_bar = self.menuBar()
        edit_menu = menu_bar.addMenu("Edit")
        self.preferences_action = edit_menu.addAction("Preferences…")
        self.preferences_action.triggered.connect(self._open_preferences)

        # Connect signals
        self.back_button.clicked.connect(self._go_back)
        self.next_button.clicked.connect(self._go_next)
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.import_panel.files_changed.connect(self._on_files_changed)
        self.analyze_panel.analysis_started.connect(self._on_analysis_started)
        self.analyze_panel.analysis_complete.connect(self._on_analysis_complete)
        self.review_panel.segments_modified.connect(self._on_segments_modified)

    def _go_back(self) -> None:
        current = self.tabs.currentIndex()
        if current > 0:
            self.tabs.setCurrentIndex(current - 1)

    def _go_next(self) -> None:
        current = self.tabs.currentIndex()
        next_idx = current + 1
        if next_idx < self.tabs.count():
            self.tabs.setTabEnabled(next_idx, True)
            self.tabs.setCurrentIndex(next_idx)

    def _on_tab_changed(self, index: int) -> None:
        self.back_button.setEnabled(index > 0)
        if index == self.tabs.count() - 1:
            self.next_button.setEnabled(False)
        elif index == 0:
            # On import tab, Next depends on whether files are loaded
            self.next_button.setEnabled(self.import_panel.file_count > 0)
        elif index == 1:
            # On analyze tab, Next requires completed analysis results
            self.next_button.setEnabled(
                self.analyze_panel.has_results and not self.analyze_panel.is_analyzing
            )
        else:
            self.next_button.setEnabled(True)

    def _on_files_changed(self) -> None:
        """Update Next button state when files are added/removed on import tab."""
        if self.tabs.currentIndex() == 0:
            self.next_button.setEnabled(self.import_panel.file_count > 0)

    def _on_analysis_started(self) -> None:
        """Disable Next while analysis is running."""
        if self.tabs.currentIndex() == 1:
            self.next_button.setEnabled(False)

    def _on_analysis_complete(self, moments: list) -> None:
        """Convert detected moments to highlight segments and pass to review."""
        segments = [
            HighlightSegment.from_moment(m, order=i) for i, m in enumerate(moments)
        ]
        segments.sort(key=lambda s: (s.source_path, s.start_time))
        self.review_panel.set_segments(segments)
        self.export_panel.set_segments(segments)
        # Re-enable Next now that results are available
        if self.tabs.currentIndex() == 1:
            self.next_button.setEnabled(True)

    def _on_segments_modified(self) -> None:
        """Sync segment changes from review to export panel."""
        self.export_panel.set_segments(self.review_panel.segments)

    def _open_preferences(self) -> None:
        """Open the settings dialog and sync changes on save."""
        dlg = SettingsDialog(self.settings, parent=self)
        if dlg.exec() == SettingsDialog.DialogCode.Accepted:
            self.analyze_panel._load_settings()
