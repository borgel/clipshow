"""DetectedMoment and HighlightSegment data models."""

from dataclasses import dataclass, field


@dataclass
class DetectedMoment:
    source_path: str
    start_time: float
    end_time: float
    peak_score: float
    mean_score: float
    contributing_detectors: list[str] = field(default_factory=list)


@dataclass
class HighlightSegment:
    source_path: str
    start_time: float
    end_time: float
    score: float
    included: bool = True
    order: int = 0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @classmethod
    def from_moment(cls, moment: DetectedMoment, order: int = 0) -> "HighlightSegment":
        return cls(
            source_path=moment.source_path,
            start_time=moment.start_time,
            end_time=moment.end_time,
            score=moment.peak_score,
            order=order,
        )
