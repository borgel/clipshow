"""Video preview widget wrapping QMediaPlayer + QVideoWidget."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QUrl, Signal, Slot
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget

logger = logging.getLogger(__name__)


class VideoPreview(QWidget):
    """Video preview with play/pause controls and segment playback."""

    playback_finished = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._segment_end: float | None = None
        self._pending_start_ms: int | None = None
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Video display
        self.video_widget = QVideoWidget()
        layout.addWidget(self.video_widget)

        # Controls
        controls = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.pause_button = QPushButton("Pause")
        self.play_button.setEnabled(False)
        self.pause_button.setEnabled(False)
        controls.addWidget(self.play_button)
        controls.addWidget(self.pause_button)
        controls.addStretch()
        layout.addLayout(controls)

        # Media player
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.player.setVideoOutput(self.video_widget)

    def _connect_signals(self) -> None:
        self.play_button.clicked.connect(self.play)
        self.pause_button.clicked.connect(self.pause)
        self.player.playbackStateChanged.connect(self._on_state_changed)
        self.player.positionChanged.connect(self._check_segment_end)
        self.player.mediaStatusChanged.connect(self._on_media_status_changed)
        self.player.errorOccurred.connect(self._on_error)

    def _on_media_status_changed(self, status: QMediaPlayer.MediaStatus) -> None:
        """Seek and play once media is loaded."""
        if status == QMediaPlayer.MediaStatus.LoadedMedia:
            if self._pending_start_ms is not None:
                self.player.setPosition(self._pending_start_ms)
                self._pending_start_ms = None
                self.player.play()

    def _on_error(self, error: QMediaPlayer.Error, message: str) -> None:
        """Handle media player errors gracefully instead of crashing."""
        logger.warning("Media player error (%s): %s", error, message)
        self.player.stop()

    def cleanup(self) -> None:
        """Stop playback and release media resources.

        Call before widget destruction to avoid segfaults from the
        underlying FFmpeg backend trying to finalize during teardown.
        """
        self.player.stop()
        self.player.setSource(QUrl())
        self.player.setVideoOutput(None)
        self.player.setAudioOutput(None)

    def closeEvent(self, event) -> None:
        self.cleanup()
        super().closeEvent(event)

    def load(self, path: str) -> None:
        """Load a video file for playback."""
        if not Path(path).is_file():
            logger.warning("Cannot load — file not found: %s", path)
            return
        self._segment_end = None
        self.player.setSource(QUrl.fromLocalFile(path))
        self.play_button.setEnabled(True)

    def play_segment(self, path: str, start_ms: int, end_ms: int) -> None:
        """Load a video and play a specific segment (start/end in milliseconds)."""
        if not Path(path).is_file():
            logger.warning("Cannot play — file not found: %s", path)
            return
        self._segment_end = end_ms
        self._pending_start_ms = start_ms
        self.player.setSource(QUrl.fromLocalFile(path))
        # Position and play are deferred to _on_media_status_changed
        # because setPosition doesn't work until media is loaded.

    @Slot()
    def play(self) -> None:
        self.player.play()

    @Slot()
    def pause(self) -> None:
        self.player.pause()

    @Slot(QMediaPlayer.PlaybackState)
    def _on_state_changed(self, state: QMediaPlayer.PlaybackState) -> None:
        is_playing = state == QMediaPlayer.PlaybackState.PlayingState
        self.play_button.setEnabled(not is_playing)
        self.pause_button.setEnabled(is_playing)
        if state == QMediaPlayer.PlaybackState.StoppedState:
            self.playback_finished.emit()

    @Slot(int)
    def _check_segment_end(self, position_ms: int) -> None:
        """Stop playback when segment end is reached."""
        if self._segment_end is not None and position_ms >= self._segment_end:
            self.player.pause()
            self._segment_end = None
            self.playback_finished.emit()

    @property
    def is_playing(self) -> bool:
        return (
            self.player.playbackState()
            == QMediaPlayer.PlaybackState.PlayingState
        )

    @property
    def position_ms(self) -> int:
        return self.player.position()

    @property
    def duration_ms(self) -> int:
        return self.player.duration()
