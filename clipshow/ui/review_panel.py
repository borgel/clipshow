"""Review panel: segment list + preview + trim controls."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from clipshow.model.moments import HighlightSegment
from clipshow.ui.segment_list import SegmentList
from clipshow.ui.video_preview import VideoPreview


class ReviewPanel(QWidget):
    """Panel for reviewing, reordering, and trimming highlight segments."""

    segments_modified = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._segments: list[HighlightSegment] = []
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Main splitter: segment list left, preview right
        splitter = QSplitter()

        self.segment_list = SegmentList()
        splitter.addWidget(self.segment_list)

        self.video_preview = VideoPreview()
        splitter.addWidget(self.video_preview)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter)

        # Trim controls
        trim_layout = QHBoxLayout()
        self.trim_start_minus = QPushButton("Start -0.5s")
        self.trim_start_plus = QPushButton("Start +0.5s")
        self.trim_end_minus = QPushButton("End -0.5s")
        self.trim_end_plus = QPushButton("End +0.5s")
        self.trim_label = QLabel("No segment selected")

        trim_layout.addWidget(self.trim_start_minus)
        trim_layout.addWidget(self.trim_start_plus)
        trim_layout.addStretch()
        trim_layout.addWidget(self.trim_label)
        trim_layout.addStretch()
        trim_layout.addWidget(self.trim_end_minus)
        trim_layout.addWidget(self.trim_end_plus)
        layout.addLayout(trim_layout)

        self._set_trim_enabled(False)

    def _connect_signals(self) -> None:
        self.segment_list.selection_changed.connect(self._on_selection_changed)
        self.segment_list.segments_modified.connect(self._on_segments_modified)
        self.trim_start_minus.clicked.connect(lambda: self._nudge_trim("start", -0.5))
        self.trim_start_plus.clicked.connect(lambda: self._nudge_trim("start", 0.5))
        self.trim_end_minus.clicked.connect(lambda: self._nudge_trim("end", -0.5))
        self.trim_end_plus.clicked.connect(lambda: self._nudge_trim("end", 0.5))

    def set_segments(self, segments: list[HighlightSegment]) -> None:
        """Load segments into the review panel."""
        self._segments = list(segments)
        self.segment_list.set_segments(self._segments)
        self._update_trim_label()

    def _on_selection_changed(self, index: int) -> None:
        self._set_trim_enabled(True)
        seg = self.segment_list.selected_segment
        if seg:
            self._update_trim_label()
            self.video_preview.play_segment(
                seg.source_path,
                int(seg.start_time * 1000),
                int(seg.end_time * 1000),
            )

    def _on_segments_modified(self) -> None:
        self._segments = self.segment_list.segments
        self.segments_modified.emit()

    def _nudge_trim(self, edge: str, delta: float) -> None:
        idx = self.segment_list.selected_index
        seg = self.segment_list.selected_segment
        if not seg:
            return
        if edge == "start":
            seg.start_time = max(0.0, seg.start_time + delta)
            # Don't let start pass end
            if seg.start_time >= seg.end_time:
                seg.start_time = seg.end_time - 0.1
        else:
            seg.end_time = max(seg.start_time + 0.1, seg.end_time + delta)
        self.segment_list._rebuild_list()
        # Restore selection after rebuild
        self.segment_list.list_widget.setCurrentRow(idx)
        self._update_trim_label()
        self.segments_modified.emit()

    def _update_trim_label(self) -> None:
        seg = self.segment_list.selected_segment
        if seg:
            self.trim_label.setText(
                f"{seg.start_time:.1f}s - {seg.end_time:.1f}s "
                f"({seg.duration:.1f}s)"
            )
        else:
            self.trim_label.setText("No segment selected")

    def _set_trim_enabled(self, enabled: bool) -> None:
        self.trim_start_minus.setEnabled(enabled)
        self.trim_start_plus.setEnabled(enabled)
        self.trim_end_minus.setEnabled(enabled)
        self.trim_end_plus.setEnabled(enabled)

    @property
    def segments(self) -> list[HighlightSegment]:
        return self.segment_list.segments

    @property
    def included_segments(self) -> list[HighlightSegment]:
        return [s for s in self.segment_list.segments if s.included]
