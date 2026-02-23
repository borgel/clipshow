"""UI tests: review panel — segment list, trim, include/exclude, preview."""

from unittest.mock import MagicMock

import pytest

from clipshow.model.moments import HighlightSegment
from clipshow.ui.review_panel import ReviewPanel
from clipshow.ui.segment_list import SegmentList


def _make_segments():
    return [
        HighlightSegment("/tmp/a.mp4", 1.0, 4.0, 0.8, included=True, order=0),
        HighlightSegment("/tmp/b.mp4", 2.0, 5.0, 0.6, included=True, order=1),
        HighlightSegment("/tmp/a.mp4", 7.0, 10.0, 0.5, included=True, order=2),
    ]


@pytest.fixture()
def panel(qtbot):
    p = ReviewPanel()
    qtbot.addWidget(p)
    return p


@pytest.fixture()
def loaded_panel(qtbot):
    p = ReviewPanel()
    qtbot.addWidget(p)
    p.set_segments(_make_segments())
    return p


class TestSegmentList:
    def test_set_segments(self, qtbot):
        sl = SegmentList()
        qtbot.addWidget(sl)
        sl.set_segments(_make_segments())
        assert sl.segment_count == 3

    def test_empty_initially(self, qtbot):
        sl = SegmentList()
        qtbot.addWidget(sl)
        assert sl.segment_count == 0

    def test_selected_segment_none(self, qtbot):
        sl = SegmentList()
        qtbot.addWidget(sl)
        assert sl.selected_segment is None

    def test_selection_signal(self, qtbot):
        sl = SegmentList()
        qtbot.addWidget(sl)
        sl.set_segments(_make_segments())
        with qtbot.waitSignal(sl.selection_changed, timeout=1000):
            sl.list_widget.setCurrentRow(1)

    def test_selected_segment(self, qtbot):
        sl = SegmentList()
        qtbot.addWidget(sl)
        sl.set_segments(_make_segments())
        sl.list_widget.setCurrentRow(1)
        seg = sl.selected_segment
        assert seg is not None
        assert seg.source_path == "/tmp/b.mp4"


class TestReviewPanelInitial:
    def test_trim_buttons_disabled(self, panel):
        assert panel.trim_start_minus.isEnabled() is False
        assert panel.trim_start_plus.isEnabled() is False
        assert panel.trim_end_minus.isEnabled() is False
        assert panel.trim_end_plus.isEnabled() is False

    def test_no_segment_label(self, panel):
        assert panel.trim_label.text() == "No segment selected"


class TestReviewPanelLoaded:
    def test_segment_count(self, loaded_panel):
        assert loaded_panel.segment_list.segment_count == 3

    def test_segments_property(self, loaded_panel):
        segs = loaded_panel.segments
        assert len(segs) == 3

    def test_all_included(self, loaded_panel):
        assert len(loaded_panel.included_segments) == 3


class TestSelection:
    def test_selecting_enables_trim(self, loaded_panel):
        loaded_panel.segment_list.list_widget.setCurrentRow(0)
        assert loaded_panel.trim_start_minus.isEnabled() is True
        assert loaded_panel.trim_end_plus.isEnabled() is True

    def test_selecting_updates_label(self, loaded_panel):
        loaded_panel.segment_list.list_widget.setCurrentRow(0)
        assert "1.0s" in loaded_panel.trim_label.text()
        assert "4.0s" in loaded_panel.trim_label.text()

    def test_selecting_triggers_preview(self, loaded_panel):
        loaded_panel.video_preview.play_segment = MagicMock()
        loaded_panel.segment_list.list_widget.setCurrentRow(1)
        loaded_panel.video_preview.play_segment.assert_called_once_with(
            "/tmp/b.mp4", 2000, 5000
        )


class TestTrimControls:
    def test_start_minus(self, loaded_panel):
        loaded_panel.segment_list.list_widget.setCurrentRow(0)
        loaded_panel.trim_start_minus.click()
        seg = loaded_panel.segment_list.segments[0]
        assert seg.start_time == pytest.approx(0.5)

    def test_start_plus(self, loaded_panel):
        loaded_panel.segment_list.list_widget.setCurrentRow(0)
        loaded_panel.trim_start_plus.click()
        seg = loaded_panel.segment_list.segments[0]
        assert seg.start_time == pytest.approx(1.5)

    def test_end_minus(self, loaded_panel):
        loaded_panel.segment_list.list_widget.setCurrentRow(0)
        loaded_panel.trim_end_minus.click()
        seg = loaded_panel.segment_list.segments[0]
        assert seg.end_time == pytest.approx(3.5)

    def test_end_plus(self, loaded_panel):
        loaded_panel.segment_list.list_widget.setCurrentRow(0)
        loaded_panel.trim_end_plus.click()
        seg = loaded_panel.segment_list.segments[0]
        assert seg.end_time == pytest.approx(4.5)

    def test_start_clamps_to_zero(self, loaded_panel):
        loaded_panel.segment_list.list_widget.setCurrentRow(0)
        # Nudge start back multiple times
        for _ in range(5):
            loaded_panel.trim_start_minus.click()
        seg = loaded_panel.segment_list.segments[0]
        assert seg.start_time >= 0.0

    def test_start_cannot_pass_end(self, loaded_panel):
        loaded_panel.segment_list.list_widget.setCurrentRow(0)
        # Push start forward past end
        for _ in range(10):
            loaded_panel.trim_start_plus.click()
        seg = loaded_panel.segment_list.segments[0]
        assert seg.start_time < seg.end_time

    def test_trim_emits_modified(self, loaded_panel, qtbot):
        loaded_panel.segment_list.list_widget.setCurrentRow(0)
        with qtbot.waitSignal(loaded_panel.segments_modified, timeout=1000):
            loaded_panel.trim_start_plus.click()

    def test_trim_updates_label(self, loaded_panel):
        loaded_panel.segment_list.list_widget.setCurrentRow(0)
        loaded_panel.trim_end_plus.click()
        assert "4.5s" in loaded_panel.trim_label.text()


class TestIncludeExclude:
    def test_uncheck_excludes(self, loaded_panel):
        loaded_panel.segment_list.list_widget.setCurrentRow(0)
        item = loaded_panel.segment_list.list_widget.item(0)
        widget = loaded_panel.segment_list.list_widget.itemWidget(item)
        widget.checkbox.setChecked(False)
        assert loaded_panel.segments[0].included is False
        assert len(loaded_panel.included_segments) == 2

    def test_recheck_includes(self, loaded_panel):
        loaded_panel.segment_list.list_widget.setCurrentRow(0)
        item = loaded_panel.segment_list.list_widget.item(0)
        widget = loaded_panel.segment_list.list_widget.itemWidget(item)
        widget.checkbox.setChecked(False)
        widget.checkbox.setChecked(True)
        assert loaded_panel.segments[0].included is True
        assert len(loaded_panel.included_segments) == 3
