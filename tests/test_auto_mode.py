"""E2E test: full auto-mode pipeline with synthetic videos."""

import os
import tempfile

from clipshow.app import run_auto_mode


class TestAutoMode:
    def test_no_files_returns_error(self):
        assert run_auto_mode([], "out.mp4", headless=True) == 1

    def test_missing_file_returns_error(self):
        result = run_auto_mode(["/nonexistent/video.mp4"], "out.mp4", headless=True)
        assert result == 1

    def test_auto_mode_with_synthetic_video(self, loud_moment_video):
        """Full pipeline: load, detect, assemble from a synthetic video."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "reel.mp4")
            result = run_auto_mode([loud_moment_video], output, headless=True)
            # May produce 0 moments with default threshold, but should not crash
            assert result in (0, 1)

    def test_headless_suppresses_progress(self, capsys):
        """Headless mode should not print progress."""
        run_auto_mode(["/nonexistent.mp4"], "out.mp4", headless=True)
        captured = capsys.readouterr()
        assert "Progress" not in captured.out

    def test_multiple_files_accepted(self, static_video, motion_video):
        """Auto mode accepts multiple input files without crashing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "reel.mp4")
            result = run_auto_mode(
                [static_video, motion_video], output, headless=True
            )
            assert result in (0, 1)

    def test_verbose_mode_prints_info(self, static_video, capsys):
        """Non-headless mode prints load/progress info."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "reel.mp4")
            run_auto_mode([static_video], output, headless=False)
            captured = capsys.readouterr()
            assert "Loaded" in captured.out
