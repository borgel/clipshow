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


class MainWindow(QMainWindow):
    """ClipShow main window with Import → Analyze → Review → Export tabs."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("ClipShow")
        self.setMinimumSize(800, 600)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Tab widget
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Create placeholder panels
        self.import_panel = QWidget()
        self.analyze_panel = QWidget()
        self.review_panel = QWidget()
        self.export_panel = QWidget()

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
        nav_layout.addStretch()
        nav_layout.addWidget(self.back_button)
        nav_layout.addWidget(self.next_button)
        layout.addLayout(nav_layout)

        # Connect signals
        self.back_button.clicked.connect(self._go_back)
        self.next_button.clicked.connect(self._go_next)
        self.tabs.currentChanged.connect(self._on_tab_changed)

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
        self.next_button.setEnabled(index < self.tabs.count() - 1)
        # Update button text on last tab
        if index == self.tabs.count() - 1:
            self.next_button.setText("Next")
            self.next_button.setEnabled(False)
        else:
            self.next_button.setText("Next")
            self.next_button.setEnabled(True)
