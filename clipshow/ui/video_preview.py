"""Video preview widget wrapping QMediaPlayer + QVideoWidget."""

from __future__ import annotations

from PySide6.QtCore import QUrl, Signal, Slot
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget


class VideoPreview(QWidget):
    """Video preview with play/pause controls and segment playback."""

    playback_finished = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._segment_end: float | None = None
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

    def load(self, path: str) -> None:
        """Load a video file for playback."""
        self._segment_end = None
        self.player.setSource(QUrl.fromLocalFile(path))
        self.play_button.setEnabled(True)

    def play_segment(self, path: str, start_ms: int, end_ms: int) -> None:
        """Load a video and play a specific segment (start/end in milliseconds)."""
        self._segment_end = end_ms
        self.player.setSource(QUrl.fromLocalFile(path))
        self.player.setPosition(start_ms)
        self.player.play()

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
