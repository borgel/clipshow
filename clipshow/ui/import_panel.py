"""Import panel: drag-drop file import + file list."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from clipshow.export.ffprobe import extract_metadata
from clipshow.model.project import Project, VideoSource

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mts"}


class DropArea(QLabel):
    """Drag-and-drop zone that accepts video files."""

    files_dropped = Signal(list)  # list[str]

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setText("Drag && drop video files here\nor click Browse")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(100)
        self.setStyleSheet(
            "QLabel { border: 2px dashed #888; border-radius: 8px; "
            "padding: 20px; color: #666; }"
        )
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = []
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if Path(path).suffix.lower() in VIDEO_EXTENSIONS:
                paths.append(path)
        if paths:
            self.files_dropped.emit(paths)


class ImportPanel(QWidget):
    """Panel for importing video files into the project."""

    files_changed = Signal()  # emitted when file list changes

    def __init__(self, project: Project, parent: QWidget | None = None):
        super().__init__(parent)
        self.project = project
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Drop area
        self.drop_area = DropArea()
        layout.addWidget(self.drop_area)

        # Browse button
        self.browse_button = QPushButton("Browse...")
        layout.addWidget(self.browse_button)

        # File table
        self.file_table = QTableWidget(0, 4)
        self.file_table.setHorizontalHeaderLabels(
            ["Filename", "Duration", "Resolution", "Path"]
        )
        self.file_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.file_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        header = self.file_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.file_table)

        # Action buttons
        btn_layout = QHBoxLayout()
        self.remove_button = QPushButton("Remove Selected")
        self.clear_button = QPushButton("Clear All")
        self.remove_button.setEnabled(False)
        self.clear_button.setEnabled(False)
        btn_layout.addWidget(self.remove_button)
        btn_layout.addWidget(self.clear_button)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    def _connect_signals(self) -> None:
        self.drop_area.files_dropped.connect(self.add_files)
        self.browse_button.clicked.connect(self._browse)
        self.remove_button.clicked.connect(self._remove_selected)
        self.clear_button.clicked.connect(self._clear_all)
        self.file_table.itemSelectionChanged.connect(self._update_button_states)

    def _browse(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Video Files",
            "",
            "Video Files (*.mp4 *.mov *.avi *.mkv *.webm *.m4v *.mts);;All Files (*)",
        )
        if paths:
            self.add_files(paths)

    def add_files(self, paths: list[str]) -> None:
        """Add video files to the project and update the table."""
        existing = {s.path for s in self.project.sources}
        for path in paths:
            if path in existing:
                continue
            try:
                source = extract_metadata(path)
            except (FileNotFoundError, RuntimeError):
                # If ffprobe fails, add with minimal info
                source = VideoSource(path=path)
            self.project.add_source(source)
            self._add_table_row(source)
            existing.add(path)
        self._update_button_states()
        self.files_changed.emit()

    def add_source_directly(self, source: VideoSource) -> None:
        """Add a pre-built VideoSource without running ffprobe."""
        existing = {s.path for s in self.project.sources}
        if source.path in existing:
            return
        self.project.add_source(source)
        self._add_table_row(source)
        self._update_button_states()
        self.files_changed.emit()

    def _add_table_row(self, source: VideoSource) -> None:
        row = self.file_table.rowCount()
        self.file_table.insertRow(row)
        self.file_table.setItem(row, 0, QTableWidgetItem(Path(source.path).name))
        self.file_table.setItem(
            row, 1, QTableWidgetItem(self._format_duration(source.duration))
        )
        self.file_table.setItem(
            row, 2, QTableWidgetItem(f"{source.width}x{source.height}")
        )
        self.file_table.setItem(row, 3, QTableWidgetItem(source.path))

    def _remove_selected(self) -> None:
        selected_rows = sorted(
            {idx.row() for idx in self.file_table.selectedIndexes()},
            reverse=True,
        )
        for row in selected_rows:
            path_item = self.file_table.item(row, 3)
            if path_item:
                self.project.remove_source(path_item.text())
            self.file_table.removeRow(row)
        self._update_button_states()
        self.files_changed.emit()

    def _clear_all(self) -> None:
        self.project.clear_sources()
        self.file_table.setRowCount(0)
        self._update_button_states()
        self.files_changed.emit()

    def _update_button_states(self) -> None:
        has_files = self.file_table.rowCount() > 0
        has_selection = len(self.file_table.selectedIndexes()) > 0
        self.remove_button.setEnabled(has_selection)
        self.clear_button.setEnabled(has_files)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        if seconds <= 0:
            return "--:--"
        m, s = divmod(int(seconds), 60)
        return f"{m}:{s:02d}"

    @property
    def file_count(self) -> int:
        return self.file_table.rowCount()
