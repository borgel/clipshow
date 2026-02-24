"""Settings dataclass with JSON persistence."""

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_CONFIG_DIR = Path.home() / ".clipshow"
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "settings.json"


@dataclass
class Settings:
    # Detector weights (0 = disabled, all off by default in UI)
    scene_weight: float = 0.0
    audio_weight: float = 0.0
    motion_weight: float = 0.0
    semantic_weight: float = 0.0
    emotion_weight: float = 0.0

    # Semantic detector prompts
    semantic_prompts: list[str] = field(
        default_factory=lambda: [
            "exciting moment",
            "people laughing",
            "beautiful scenery",
        ]
    )
    semantic_negative_prompts: list[str] = field(
        default_factory=lambda: [
            "boring static shot",
            "blank wall",
            "empty room",
            "black screen",
        ]
    )

    # Segment selection
    pre_padding_sec: float = 1.0
    post_padding_sec: float = 1.5
    score_threshold: float = 0.5
    min_segment_duration_sec: float = 1.0
    max_segment_duration_sec: float = 15.0

    # Parallelism
    max_workers: int = 0  # 0 = auto (cpu_count)

    # Output
    output_codec: str = "libx264"
    output_fps: float = 30.0
    output_bitrate: str = "8M"

    def save(self, path: Path | None = None) -> None:
        path = path or DEFAULT_CONFIG_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path | None = None) -> "Settings":
        path = path or DEFAULT_CONFIG_PATH
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text())
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        except (json.JSONDecodeError, TypeError):
            return cls()

    @property
    def resolved_max_workers(self) -> int:
        if self.max_workers > 0:
            return self.max_workers
        return os.cpu_count() or 4

    @property
    def detector_weights(self) -> dict[str, float]:
        return {
            "scene": self.scene_weight,
            "audio": self.audio_weight,
            "motion": self.motion_weight,
            "semantic": self.semantic_weight,
            "emotion": self.emotion_weight,
        }

    @property
    def enabled_detectors(self) -> list[str]:
        return [name for name, weight in self.detector_weights.items() if weight > 0]
