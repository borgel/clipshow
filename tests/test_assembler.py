"""Integration tests for video assembler."""

import os

import pytest

from clipshow.export.assembler import assemble_highlights
from clipshow.export.ffprobe import extract_metadata
from clipshow.model.moments import HighlightSegment


class TestAssembleHighlights:
    def test_concatenate_two_clips(self, static_video, scene_change_video, tmp_path):
        """Assembling two full clips should produce a valid output."""
        output = str(tmp_path / "output.mp4")
        segments = [
            HighlightSegment(
                source_path=static_video,
                start_time=0.0,
                end_time=1.0,
                score=0.8,
                included=True,
                order=0,
            ),
            HighlightSegment(
                source_path=scene_change_video,
                start_time=0.0,
                end_time=1.0,
                score=0.9,
                included=True,
                order=1,
            ),
        ]
        result = assemble_highlights(segments, output)
        assert result == output
        assert os.path.exists(output)

        # Verify output is valid video
        meta = extract_metadata(output)
        assert meta.codec == "h264"
        assert meta.duration == pytest.approx(2.0, abs=0.3)

    def test_subclip_times(self, static_video, tmp_path):
        """Subclip start/end should be respected."""
        output = str(tmp_path / "subclip.mp4")
        segments = [
            HighlightSegment(
                source_path=static_video,
                start_time=0.2,
                end_time=0.8,
                score=0.7,
                included=True,
                order=0,
            ),
        ]
        result = assemble_highlights(segments, output)
        assert os.path.exists(result)

        meta = extract_metadata(result)
        assert meta.duration == pytest.approx(0.6, abs=0.2)

    def test_excluded_segments_skipped(self, static_video, tmp_path):
        """Segments with included=False should be skipped."""
        output = str(tmp_path / "filtered.mp4")
        segments = [
            HighlightSegment(
                source_path=static_video,
                start_time=0.0,
                end_time=1.0,
                score=0.5,
                included=False,  # Excluded
                order=0,
            ),
            HighlightSegment(
                source_path=static_video,
                start_time=0.0,
                end_time=1.0,
                score=0.8,
                included=True,
                order=1,
            ),
        ]
        result = assemble_highlights(segments, output)
        assert os.path.exists(result)

        meta = extract_metadata(result)
        # Only one 1s segment included
        assert meta.duration == pytest.approx(1.0, abs=0.3)

    def test_no_segments_raises(self, tmp_path):
        output = str(tmp_path / "empty.mp4")
        with pytest.raises(ValueError, match="No included segments"):
            assemble_highlights([], output)

    def test_all_excluded_raises(self, static_video, tmp_path):
        output = str(tmp_path / "all_excluded.mp4")
        segments = [
            HighlightSegment(
                source_path=static_video,
                start_time=0.0,
                end_time=1.0,
                score=0.5,
                included=False,
                order=0,
            ),
        ]
        with pytest.raises(ValueError, match="No included segments"):
            assemble_highlights(segments, output)

    def test_progress_callback(self, static_video, tmp_path):
        output = str(tmp_path / "progress.mp4")
        segments = [
            HighlightSegment(
                source_path=static_video,
                start_time=0.0,
                end_time=1.0,
                score=0.8,
                included=True,
                order=0,
            ),
        ]
        progress_values = []
        assemble_highlights(segments, output, progress_callback=progress_values.append)
        assert len(progress_values) > 0
        assert progress_values[-1] == pytest.approx(1.0)
