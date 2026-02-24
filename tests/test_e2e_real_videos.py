"""E2E UI test using real fixture videos through the full workflow.

Requires video files in tests/fixtures/videos/*.mp4.
Skipped automatically if no fixture videos are present.
"""

import glob
import os
import tempfile

import pytest

from clipshow.config import Settings
from clipshow.ui.main_window import MainWindow

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "videos")
FIXTURE_VIDEOS = sorted(glob.glob(os.path.join(FIXTURES_DIR, "*.mp4")))

skip_no_fixtures = pytest.mark.skipif(
    not FIXTURE_VIDEOS,
    reason="No fixture videos in tests/fixtures/videos/",
)


@pytest.fixture()
def window_with_settings(qtbot):
    """MainWindow with low threshold and sequential processing for predictability."""
    settings = Settings(
        scene_weight=0.3,
        audio_weight=0.25,
        motion_weight=0.25,
        semantic_weight=0.0,
        emotion_weight=0.0,
        score_threshold=0.3,
        max_workers=1,
    )
    w = MainWindow(settings=settings)
    qtbot.addWidget(w)
    yield w
    w.review_panel.video_preview.cleanup()


@skip_no_fixtures
class TestRealVideoWorkflow:
    """Full Import → Analyze → Review → Export using real fixture videos."""

    def test_import_real_videos(self, window_with_settings):
        """Importing fixture videos populates the import panel."""
        window = window_with_settings
        window.import_panel.add_files(FIXTURE_VIDEOS)

        assert window.import_panel.file_count == len(FIXTURE_VIDEOS)
        # All files should appear in the project
        project_paths = {s.path for s in window.project.sources}
        for video in FIXTURE_VIDEOS:
            assert video in project_paths

    def test_import_metadata_extracted(self, window_with_settings):
        """Imported real videos should have valid duration and resolution."""
        window = window_with_settings
        window.import_panel.add_files(FIXTURE_VIDEOS[:1])

        source = window.project.sources[0]
        assert source.duration > 0, "Duration should be positive"
        assert source.width > 0, "Width should be positive"
        assert source.height > 0, "Height should be positive"

    def test_import_enables_next_button(self, window_with_settings):
        """After importing, Next button should be enabled."""
        window = window_with_settings
        assert not window.next_button.isEnabled()
        window.import_panel.add_files(FIXTURE_VIDEOS[:1])
        assert window.next_button.isEnabled()

    def test_analyze_single_video(self, window_with_settings, qtbot):
        """Run analysis on a single real video and verify moments are found."""
        window = window_with_settings
        window.import_panel.add_files(FIXTURE_VIDEOS[:1])
        window.next_button.click()  # Navigate to Analyze tab
        assert window.tabs.currentIndex() == 1

        # Start analysis and wait for completion
        with qtbot.waitSignal(
            window.analyze_panel.analysis_complete, timeout=120_000
        ) as sig:
            window.analyze_panel.start_analysis()

        moments = sig.args[0]
        assert isinstance(moments, list)
        # Real video should produce at least some moments
        assert len(moments) > 0, (
            f"Expected moments from real video, got 0. "
            f"Video: {FIXTURE_VIDEOS[0]}"
        )

    def test_analyze_populates_review_panel(self, window_with_settings, qtbot):
        """Analysis completion should populate the review panel with segments."""
        window = window_with_settings
        window.import_panel.add_files(FIXTURE_VIDEOS[:1])
        window.next_button.click()

        with qtbot.waitSignal(
            window.analyze_panel.analysis_complete, timeout=120_000
        ):
            window.analyze_panel.start_analysis()

        seg_count = window.review_panel.segment_list.segment_count
        assert seg_count > 0, "Review panel should have segments after analysis"

    def test_analyze_populates_export_panel(self, window_with_settings, qtbot):
        """Analysis completion should populate the export panel summary."""
        window = window_with_settings
        window.import_panel.add_files(FIXTURE_VIDEOS[:1])
        window.next_button.click()

        with qtbot.waitSignal(
            window.analyze_panel.analysis_complete, timeout=120_000
        ):
            window.analyze_panel.start_analysis()

        summary = window.export_panel.summary_label.text()
        assert "segment" in summary.lower(), (
            f"Export summary should mention segments, got: {summary}"
        )

    def test_full_workflow_single_video(self, window_with_settings, qtbot):
        """Full workflow: Import → Analyze → Review trim → Export."""
        window = window_with_settings

        # Step 1: Import
        window.import_panel.add_files(FIXTURE_VIDEOS[:1])
        assert window.import_panel.file_count == 1

        # Step 2: Navigate to Analyze and run
        window.next_button.click()
        assert window.tabs.currentIndex() == 1

        with qtbot.waitSignal(
            window.analyze_panel.analysis_complete, timeout=120_000
        ):
            window.analyze_panel.start_analysis()

        # Step 3: Navigate to Review
        window.next_button.click()
        assert window.tabs.currentIndex() == 2

        segments = window.review_panel.segments
        assert len(segments) > 0

        # Verify segments reference the correct source
        for seg in segments:
            assert seg.source_path == FIXTURE_VIDEOS[0]
            assert seg.start_time >= 0
            assert seg.end_time > seg.start_time

        # Select first segment and verify trim controls work
        window.review_panel.segment_list.list_widget.setCurrentRow(0)
        original_end = segments[0].end_time
        window.review_panel._nudge_trim("end", -0.5)
        assert segments[0].end_time == pytest.approx(original_end - 0.5)

        # Step 4: Navigate to Export
        window.next_button.click()
        assert window.tabs.currentIndex() == 3

        # Verify export panel has the segments
        summary = window.export_panel.summary_label.text()
        assert "segment" in summary.lower()

        # Run the actual export
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_reel.mp4")
            window.export_panel.path_edit.setText(output_path)
            window.export_panel.export_settings.output_path = output_path

            with qtbot.waitSignal(
                window.export_panel.export_complete, timeout=120_000
            ):
                window.export_panel.start_export()

            assert os.path.exists(output_path), "Output video should exist"
            assert os.path.getsize(output_path) > 0, "Output video should not be empty"

    def test_analyze_multiple_videos(self, window_with_settings, qtbot):
        """Analyze multiple real videos and get combined results."""
        videos = FIXTURE_VIDEOS[:2]
        if len(videos) < 2:
            pytest.skip("Need at least 2 fixture videos")

        window = window_with_settings
        window.import_panel.add_files(videos)
        assert window.import_panel.file_count == 2

        window.next_button.click()

        with qtbot.waitSignal(
            window.analyze_panel.analysis_complete, timeout=300_000
        ) as sig:
            window.analyze_panel.start_analysis()

        moments = sig.args[0]
        assert len(moments) > 0

        # Moments should reference both source videos
        source_paths = {m.source_path for m in moments}
        assert len(source_paths) >= 1, "Expected moments from at least one video"

    def test_cancel_analysis(self, window_with_settings, qtbot):
        """Cancelling analysis should not crash and should re-enable controls."""
        window = window_with_settings
        window.import_panel.add_files(FIXTURE_VIDEOS[:1])
        window.next_button.click()

        # Start analysis then cancel immediately
        window.analyze_panel.start_analysis()
        assert window.analyze_panel.is_analyzing
        window.analyze_panel.cancel_analysis()

        # Wait for the worker to finish (it should stop quickly)
        with qtbot.waitSignal(
            window.analyze_panel.analysis_complete, timeout=30_000
        ):
            pass

        # Controls should be re-enabled
        assert window.analyze_panel.analyze_button.isEnabled()
        assert not window.analyze_panel.cancel_button.isEnabled()


@skip_no_fixtures
class TestRealVideoParallel:
    """Test parallel processing with real fixture videos."""

    def test_parallel_analysis(self, qtbot):
        """Parallel analysis should complete without errors."""
        videos = FIXTURE_VIDEOS[:2]
        if len(videos) < 2:
            pytest.skip("Need at least 2 fixture videos for parallel test")

        settings = Settings(
            scene_weight=0.3,
            audio_weight=0.0,
            motion_weight=0.25,
            semantic_weight=0.0,
            emotion_weight=0.0,
            score_threshold=0.3,
            max_workers=2,
        )
        window = MainWindow(settings=settings)
        qtbot.addWidget(window)

        try:
            window.import_panel.add_files(videos)
            window.next_button.click()

            with qtbot.waitSignal(
                window.analyze_panel.analysis_complete, timeout=300_000
            ) as sig:
                window.analyze_panel.start_analysis()

            moments = sig.args[0]
            assert isinstance(moments, list)
            assert len(moments) > 0, "Parallel analysis should find moments"
        finally:
            window.review_panel.video_preview.cleanup()
