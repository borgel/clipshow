"""Abstract Detector base class and DetectorResult."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class DetectorResult:
    """Result from a single detector run on a single video.

    Attributes:
        name: Detector name (e.g. "scene", "audio", "motion").
        scores: 1D numpy array of scores, one per timestep, normalized to [0, 1].
        time_step: Time interval between consecutive scores in seconds.
        source_path: Path to the video file that was analyzed.
    """

    name: str
    scores: np.ndarray
    time_step: float
    source_path: str

    @property
    def duration(self) -> float:
        return len(self.scores) * self.time_step

    @property
    def num_samples(self) -> int:
        return len(self.scores)


class Detector(ABC):
    """Abstract base class for all detectors."""

    name: str = "base"

    @abstractmethod
    def detect(
        self,
        video_path: str,
        progress_callback: callable | None = None,
        cancel_flag: callable | None = None,
    ) -> DetectorResult:
        """Run detection on a video file.

        Args:
            video_path: Path to the video file.
            progress_callback: Optional callable(float) for progress updates (0-1).
            cancel_flag: Optional callable() -> bool that returns True to cancel.

        Returns:
            DetectorResult with normalized scores.
        """
        ...
