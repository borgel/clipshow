"""LanguageBind-based audio-visual-text detector.

Jointly scores video frames and audio against text prompts using
LanguageBind's shared embedding space. Unlike SemanticDetector (which
only sees video) and AudioDetector (which only sees audio), this detector
understands cross-modal semantics: a cheering crowd sounds exciting AND
looks exciting, and the model was trained to fuse those signals.

Stub — tests define the interface (TDD red phase).
Implementation in clipshow-7lq will make tests pass (green).
"""

from __future__ import annotations

from clipshow.detection.audio_preprocess import (  # noqa: F401
    extract_mel_windows,
    format_for_languagebind,
)
from clipshow.detection.base import Detector, DetectorResult
from clipshow.detection.models import ModelManager  # noqa: F401

DEFAULT_PROMPTS = [
    "exciting moment",
    "people laughing",
    "beautiful scenery",
    "action scene",
]
DEFAULT_NEGATIVE_PROMPTS = [
    "boring static shot",
    "blank wall",
    "empty room",
    "black screen",
]

# Video frame sampling rate
SAMPLE_FPS = 2


class AudioVisualDetector(Detector):
    """LanguageBind-based audio-visual-text detector."""

    name = "audiovisual"

    def __init__(
        self,
        prompts: list[str] | None = None,
        negative_prompts: list[str] | None = None,
        audio_weight: float = 0.4,
        time_step: float = 0.1,
    ):
        self._prompts = prompts or DEFAULT_PROMPTS
        self._negative_prompts = negative_prompts or DEFAULT_NEGATIVE_PROMPTS
        self._audio_weight = audio_weight
        self._time_step = time_step
        self._video_session = None
        self._audio_session = None
        self._text_session = None

    def _load_models(self, progress_callback=None):
        """Load all three LanguageBind ONNX sessions."""
        raise NotImplementedError

    def detect(
        self,
        video_path: str,
        progress_callback: callable | None = None,
        cancel_flag: callable | None = None,
    ) -> DetectorResult:
        raise NotImplementedError
