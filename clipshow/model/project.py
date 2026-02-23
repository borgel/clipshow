"""Project, VideoSource, and ExportSettings data models."""

from dataclasses import dataclass, field


@dataclass
class VideoSource:
    path: str
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    codec: str = ""


@dataclass
class ExportSettings:
    output_path: str = "highlight_reel.mp4"
    codec: str = "libx264"
    fps: float = 30.0
    bitrate: str = "8M"


@dataclass
class Project:
    sources: list[VideoSource] = field(default_factory=list)
    export_settings: ExportSettings = field(default_factory=ExportSettings)

    def add_source(self, source: VideoSource) -> None:
        self.sources.append(source)

    def remove_source(self, path: str) -> None:
        self.sources = [s for s in self.sources if s.path != path]

    def clear_sources(self) -> None:
        self.sources.clear()
