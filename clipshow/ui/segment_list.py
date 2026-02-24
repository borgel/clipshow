"""Draggable reorderable segment list widget."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from clipshow.model.moments import HighlightSegment


class SegmentItemWidget(QWidget):
    """Custom widget for a single segment row in the list."""

    toggled = Signal(int, bool)  # index, included

    def __init__(self, segment: HighlightSegment, index: int, parent=None):
        super().__init__(parent)
        self.segment = segment
        self.index = index

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)

        self.checkbox = QCheckBox()
        self.checkbox.setChecked(segment.included)
        self.checkbox.toggled.connect(self._on_toggled)
        layout.addWidget(self.checkbox)

        from pathlib import Path

        name = Path(segment.source_path).name
        text = (
            f"{name}  {segment.start_time:.1f}s - {segment.end_time:.1f}s  "
            f"(score: {segment.score:.2f})"
        )
        if segment.detectors:
            tags = ", ".join(d.capitalize() for d in segment.detectors)
            text += f"  [{tags}]"
        self.label = QLabel(text)
        layout.addWidget(self.label)
        layout.addStretch()

    def _on_toggled(self, checked: bool) -> None:
        self.segment.included = checked
        self.toggled.emit(self.index, checked)


class SegmentList(QWidget):
    """List of highlight segments with drag-to-reorder and include/exclude."""

    selection_changed = Signal(int)  # index of selected segment
    segments_modified = Signal()  # any change to ordering or inclusion

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._segments: list[HighlightSegment] = []
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.list_widget = QListWidget()
        self.list_widget.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.list_widget.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.list_widget.currentRowChanged.connect(self._on_row_changed)
        self.list_widget.model().rowsMoved.connect(self._on_rows_moved)
        layout.addWidget(self.list_widget)

    def set_segments(self, segments: list[HighlightSegment]) -> None:
        """Replace all segments in the list."""
        self._segments = list(segments)
        self._rebuild_list()

    def _rebuild_list(self) -> None:
        self.list_widget.clear()
        for i, seg in enumerate(self._segments):
            item = QListWidgetItem()
            widget = SegmentItemWidget(seg, i)
            widget.toggled.connect(self._on_toggled)
            item.setSizeHint(widget.sizeHint())
            self.list_widget.addItem(item)
            self.list_widget.setItemWidget(item, widget)

    def _on_row_changed(self, row: int) -> None:
        if row >= 0:
            self.selection_changed.emit(row)

    def _on_rows_moved(self) -> None:
        # Reorder internal segment list to match visual order
        new_segments = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            widget = self.list_widget.itemWidget(item)
            if widget:
                new_segments.append(widget.segment)
        self._segments = new_segments
        self.segments_modified.emit()

    def _on_toggled(self, index: int, included: bool) -> None:
        self.segments_modified.emit()

    @property
    def segments(self) -> list[HighlightSegment]:
        return list(self._segments)

    @property
    def selected_index(self) -> int:
        return self.list_widget.currentRow()

    @property
    def selected_segment(self) -> HighlightSegment | None:
        idx = self.selected_index
        if 0 <= idx < len(self._segments):
            return self._segments[idx]
        return None

    @property
    def segment_count(self) -> int:
        return len(self._segments)
