"""Reorderable segment table widget."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from clipshow.model.moments import HighlightSegment

# Column indices
COL_INCLUDE = 0
COL_FILE = 1
COL_START = 2
COL_END = 3
COL_DURATION = 4
COL_SCORE = 5
COL_DETECTORS = 6
_COLUMN_HEADERS = ["", "File", "Start", "End", "Duration", "Score", "Detectors"]


class SegmentList(QWidget):
    """Table of highlight segments with drag-to-reorder and include/exclude."""

    selection_changed = Signal(int)  # index of selected segment
    segments_modified = Signal()  # any change to ordering or inclusion

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._segments: list[HighlightSegment] = []
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.table_widget = QTableWidget(0, len(_COLUMN_HEADERS))
        self.table_widget.setHorizontalHeaderLabels(_COLUMN_HEADERS)
        self.table_widget.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table_widget.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.table_widget.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.table_widget.setDragDropMode(
            QAbstractItemView.DragDropMode.InternalMove
        )
        self.table_widget.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.table_widget.verticalHeader().setVisible(False)

        # Column sizing
        header = self.table_widget.horizontalHeader()
        header.setSectionResizeMode(COL_INCLUDE, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(COL_FILE, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(COL_START, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(COL_END, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(COL_DURATION, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(COL_SCORE, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(COL_DETECTORS, QHeaderView.ResizeMode.Stretch)

        self.table_widget.currentCellChanged.connect(self._on_cell_changed)
        self.table_widget.model().rowsMoved.connect(self._on_rows_moved)

        layout.addWidget(self.table_widget)

    def set_segments(self, segments: list[HighlightSegment]) -> None:
        """Replace all segments in the table."""
        self._segments = list(segments)
        self._rebuild_list()

    def _rebuild_list(self) -> None:
        self.table_widget.setRowCount(0)
        self.table_widget.setRowCount(len(self._segments))
        for i, seg in enumerate(self._segments):
            self._populate_row(i, seg)

    def _populate_row(self, row: int, seg: HighlightSegment) -> None:
        # Include checkbox
        cb = QCheckBox()
        cb.setChecked(seg.included)
        cb.toggled.connect(lambda checked, r=row: self._on_toggled(r, checked))
        self.table_widget.setCellWidget(row, COL_INCLUDE, cb)

        # File name
        name = Path(seg.source_path).name
        self.table_widget.setItem(row, COL_FILE, QTableWidgetItem(name))

        # Start / End / Duration
        self.table_widget.setItem(
            row, COL_START, QTableWidgetItem(f"{seg.start_time:.1f}s")
        )
        self.table_widget.setItem(
            row, COL_END, QTableWidgetItem(f"{seg.end_time:.1f}s")
        )
        self.table_widget.setItem(
            row, COL_DURATION, QTableWidgetItem(f"{seg.duration:.1f}s")
        )

        # Score
        self.table_widget.setItem(
            row, COL_SCORE, QTableWidgetItem(f"{seg.score:.2f}")
        )

        # Detectors
        if seg.detectors:
            tags = ", ".join(d.capitalize() for d in seg.detectors)
        else:
            tags = ""
        self.table_widget.setItem(row, COL_DETECTORS, QTableWidgetItem(tags))

    def _on_cell_changed(self, row: int, _col: int, _prev_row: int, _prev_col: int) -> None:
        if row >= 0:
            self.selection_changed.emit(row)

    def _on_rows_moved(self) -> None:
        # Reorder internal segment list to match visual order
        # After a drag, the table rows are rearranged; rebuild from cell widgets
        new_segments = []
        for i in range(self.table_widget.rowCount()):
            file_item = self.table_widget.item(i, COL_FILE)
            if file_item:
                # Find matching segment by file + start time text
                start_text = self.table_widget.item(i, COL_START).text()
                for seg in self._segments:
                    name = Path(seg.source_path).name
                    if name == file_item.text() and f"{seg.start_time:.1f}s" == start_text:
                        new_segments.append(seg)
                        break
        if len(new_segments) == len(self._segments):
            self._segments = new_segments
        self.segments_modified.emit()

    def _on_toggled(self, row: int, included: bool) -> None:
        if 0 <= row < len(self._segments):
            self._segments[row].included = included
        self.segments_modified.emit()

    @property
    def segments(self) -> list[HighlightSegment]:
        return list(self._segments)

    @property
    def selected_index(self) -> int:
        return self.table_widget.currentRow()

    @property
    def selected_segment(self) -> HighlightSegment | None:
        idx = self.selected_index
        if 0 <= idx < len(self._segments):
            return self._segments[idx]
        return None

    @property
    def segment_count(self) -> int:
        return len(self._segments)

    def include_checkbox(self, row: int) -> QCheckBox | None:
        """Return the include checkbox widget at the given row."""
        widget = self.table_widget.cellWidget(row, COL_INCLUDE)
        if isinstance(widget, QCheckBox):
            return widget
        return None
