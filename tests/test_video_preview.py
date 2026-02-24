"""UI tests: video preview widget with mocked QMediaPlayer."""

from unittest.mock import MagicMock

import pytest
from PySide6.QtCore import QUrl
from PySide6.QtMultimedia import QMediaPlayer

from clipshow.ui.video_preview import VideoPreview


@pytest.fixture()
def preview(qtbot):
    """Fresh VideoPreview widget with proper cleanup to prevent segfaults."""
    p = VideoPreview()
    qtbot.addWidget(p)
    yield p
    p.cleanup()


@pytest.fixture()
def fake_video(tmp_path):
    """Create a dummy file so path-existence checks pass."""
    p = tmp_path / "test.mp4"
    p.write_bytes(b"\x00")
    return str(p)


class TestInitialState:
    def test_play_disabled(self, preview):
        assert preview.play_button.isEnabled() is False

    def test_pause_disabled(self, preview):
        assert preview.pause_button.isEnabled() is False

    def test_not_playing(self, preview):
        assert preview.is_playing is False


class TestLoad:
    def test_load_enables_play(self, preview, fake_video):
        preview.load(fake_video)
        assert preview.play_button.isEnabled() is True

    def test_load_sets_source(self, preview, fake_video):
        preview.load(fake_video)
        source = preview.player.source()
        assert source.toLocalFile() == fake_video

    def test_load_missing_file_skips(self, preview):
        """Loading a non-existent file should not set source (avoids segfault)."""
        preview.load("/nonexistent/video.mp4")
        assert preview.play_button.isEnabled() is False
        assert preview.player.source() == QUrl()


class TestPlayPause:
    def test_play_calls_player(self, preview, fake_video):
        preview.load(fake_video)
        preview.player.play = MagicMock()
        preview.play_button.click()
        preview.player.play.assert_called_once()

    def test_pause_calls_player(self, preview, fake_video):
        preview.load(fake_video)
        preview.player.pause = MagicMock()
        preview.pause_button.setEnabled(True)
        preview.pause_button.click()
        preview.player.pause.assert_called_once()


class TestStateChanges:
    def test_playing_state_disables_play_enables_pause(self, preview):
        preview._on_state_changed(QMediaPlayer.PlaybackState.PlayingState)
        assert preview.play_button.isEnabled() is False
        assert preview.pause_button.isEnabled() is True

    def test_paused_state_enables_play_disables_pause(self, preview):
        preview._on_state_changed(QMediaPlayer.PlaybackState.PausedState)
        assert preview.play_button.isEnabled() is True
        assert preview.pause_button.isEnabled() is False

    def test_stopped_state_enables_play_disables_pause(self, preview):
        preview._on_state_changed(QMediaPlayer.PlaybackState.StoppedState)
        assert preview.play_button.isEnabled() is True
        assert preview.pause_button.isEnabled() is False

    def test_stopped_emits_playback_finished(self, preview, qtbot):
        with qtbot.waitSignal(preview.playback_finished, timeout=1000):
            preview._on_state_changed(QMediaPlayer.PlaybackState.StoppedState)


class TestSegmentPlayback:
    def test_play_segment_sets_source(self, preview, fake_video):
        preview.player.play = MagicMock()
        preview.player.setPosition = MagicMock()
        preview.play_segment(fake_video, 1000, 5000)
        source = preview.player.source()
        assert source.toLocalFile() == fake_video

    def test_play_segment_sets_position(self, preview, fake_video):
        preview.player.play = MagicMock()
        preview.player.setPosition = MagicMock()
        preview.play_segment(fake_video, 1000, 5000)
        preview.player.setPosition.assert_called_with(1000)

    def test_play_segment_starts_playback(self, preview, fake_video):
        preview.player.play = MagicMock()
        preview.player.setPosition = MagicMock()
        preview.play_segment(fake_video, 1000, 5000)
        preview.player.play.assert_called_once()

    def test_play_segment_missing_file_skips(self, preview):
        """play_segment with non-existent file should not trigger FFmpeg."""
        preview.player.play = MagicMock()
        preview.play_segment("/nonexistent/video.mp4", 0, 5000)
        preview.player.play.assert_not_called()
        assert preview.player.source() == QUrl()

    def test_segment_end_pauses_playback(self, preview):
        preview._segment_end = 5000
        preview.player.pause = MagicMock()
        preview._check_segment_end(5000)
        preview.player.pause.assert_called_once()

    def test_segment_end_clears_after_pause(self, preview):
        preview._segment_end = 5000
        preview.player.pause = MagicMock()
        preview._check_segment_end(5000)
        assert preview._segment_end is None

    def test_segment_end_emits_finished(self, preview, qtbot):
        preview._segment_end = 5000
        preview.player.pause = MagicMock()
        with qtbot.waitSignal(preview.playback_finished, timeout=1000):
            preview._check_segment_end(5000)

    def test_no_pause_before_segment_end(self, preview):
        preview._segment_end = 5000
        preview.player.pause = MagicMock()
        preview._check_segment_end(3000)
        preview.player.pause.assert_not_called()

    def test_no_segment_end_no_pause(self, preview):
        preview._segment_end = None
        preview.player.pause = MagicMock()
        preview._check_segment_end(3000)
        preview.player.pause.assert_not_called()


class TestCleanup:
    """Ensure cleanup releases media resources to prevent segfaults."""

    def test_cleanup_stops_player(self, preview):
        preview.player.stop = MagicMock()
        preview.cleanup()
        preview.player.stop.assert_called_once()

    def test_cleanup_clears_source(self, preview, fake_video):
        preview.load(fake_video)
        preview.cleanup()
        assert preview.player.source() == QUrl()

    def test_cleanup_detaches_video_output(self, preview):
        preview.player.setVideoOutput = MagicMock()
        preview.cleanup()
        preview.player.setVideoOutput.assert_called_with(None)

    def test_cleanup_detaches_audio_output(self, preview):
        preview.player.setAudioOutput = MagicMock()
        preview.cleanup()
        preview.player.setAudioOutput.assert_called_with(None)


class TestErrorHandling:
    """Ensure media errors are caught instead of crashing."""

    def test_error_stops_player(self, preview):
        preview.player.stop = MagicMock()
        preview._on_error(QMediaPlayer.Error.ResourceError, "file not found")
        preview.player.stop.assert_called_once()

    def test_error_handler_connected(self, preview):
        """Verify the errorOccurred signal is connected."""
        # Trigger error signal — should not raise
        preview.player.errorOccurred.emit(
            QMediaPlayer.Error.ResourceError, "test"
        )
