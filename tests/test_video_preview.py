"""UI tests: video preview widget with mocked QMediaPlayer."""

from unittest.mock import MagicMock

import pytest
from PySide6.QtMultimedia import QMediaPlayer

from clipshow.ui.video_preview import VideoPreview


@pytest.fixture()
def preview(qtbot):
    """Fresh VideoPreview widget."""
    p = VideoPreview()
    qtbot.addWidget(p)
    return p


class TestInitialState:
    def test_play_disabled(self, preview):
        assert preview.play_button.isEnabled() is False

    def test_pause_disabled(self, preview):
        assert preview.pause_button.isEnabled() is False

    def test_not_playing(self, preview):
        assert preview.is_playing is False


class TestLoad:
    def test_load_enables_play(self, preview):
        preview.load("/tmp/test.mp4")
        assert preview.play_button.isEnabled() is True

    def test_load_sets_source(self, preview):
        preview.load("/tmp/test.mp4")
        source = preview.player.source()
        assert source.toLocalFile() == "/tmp/test.mp4"


class TestPlayPause:
    def test_play_calls_player(self, preview):
        preview.load("/tmp/test.mp4")
        preview.player.play = MagicMock()
        preview.play_button.click()
        preview.player.play.assert_called_once()

    def test_pause_calls_player(self, preview):
        preview.load("/tmp/test.mp4")
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
    def test_play_segment_sets_source(self, preview):
        preview.player.play = MagicMock()
        preview.player.setPosition = MagicMock()
        preview.play_segment("/tmp/test.mp4", 1000, 5000)
        source = preview.player.source()
        assert source.toLocalFile() == "/tmp/test.mp4"

    def test_play_segment_sets_position(self, preview):
        preview.player.play = MagicMock()
        preview.player.setPosition = MagicMock()
        preview.play_segment("/tmp/test.mp4", 1000, 5000)
        preview.player.setPosition.assert_called_with(1000)

    def test_play_segment_starts_playback(self, preview):
        preview.player.play = MagicMock()
        preview.player.setPosition = MagicMock()
        preview.play_segment("/tmp/test.mp4", 1000, 5000)
        preview.player.play.assert_called_once()

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
