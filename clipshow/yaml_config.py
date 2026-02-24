"""YAML pipeline configuration for batch/CLI usage."""

from __future__ import annotations

import glob
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from clipshow.config import Settings

KNOWN_TOP_KEYS = {"inputs", "output", "detectors", "semantic", "segments", "workers"}
KNOWN_OUTPUT_KEYS = {"path", "codec", "fps", "bitrate"}
KNOWN_SEGMENT_KEYS = {"threshold", "min_duration", "max_duration", "pre_padding", "post_padding"}
KNOWN_SEMANTIC_KEYS = {"prompts", "negative_prompts"}


@dataclass
class PipelineConfig:
    """All fields are None by default — unset means 'use the default'."""

    inputs: list[str] = field(default_factory=list)

    # Output
    output_path: str | None = None
    output_codec: str | None = None
    output_fps: float | None = None
    output_bitrate: str | None = None

    # Detector weights
    scene_weight: float | None = None
    audio_weight: float | None = None
    motion_weight: float | None = None
    semantic_weight: float | None = None
    emotion_weight: float | None = None

    # Semantic prompts
    semantic_prompts: list[str] | None = None
    semantic_negative_prompts: list[str] | None = None

    # Segment selection
    score_threshold: float | None = None
    min_segment_duration: float | None = None
    max_segment_duration: float | None = None
    pre_padding: float | None = None
    post_padding: float | None = None

    # Parallelism
    workers: int | None = None


def _expand_globs(patterns: list[str], base_dir: Path) -> list[str]:
    """Expand glob patterns relative to *base_dir*, returning unique absolute paths."""
    seen: set[str] = set()
    results: list[str] = []
    for pattern in patterns:
        # Resolve relative patterns against the YAML file's directory
        if not Path(pattern).is_absolute():
            pattern = str(base_dir / pattern)
        matches = sorted(glob.glob(pattern))
        for m in matches:
            resolved = str(Path(m).resolve())
            if resolved not in seen:
                seen.add(resolved)
                results.append(resolved)
    return results


def _warn_unknown_keys(keys: set[str], known: set[str], section: str) -> None:
    unknown = keys - known
    for key in sorted(unknown):
        warnings.warn(f"Unknown key '{key}' in {section} section of pipeline config", stacklevel=3)


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    """Load a YAML pipeline config file and return a PipelineConfig."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = yaml.safe_load(path.read_text())
    if raw is None:
        # Empty YAML file
        return PipelineConfig()
    if not isinstance(raw, dict):
        raise ValueError(f"Pipeline config must be a YAML mapping, got {type(raw).__name__}")

    _warn_unknown_keys(set(raw.keys()), KNOWN_TOP_KEYS, "top-level")

    base_dir = path.resolve().parent
    cfg = PipelineConfig()

    # --- inputs ---
    if "inputs" in raw:
        patterns = raw["inputs"]
        if isinstance(patterns, str):
            patterns = [patterns]
        if not isinstance(patterns, list):
            raise ValueError("'inputs' must be a string or list of strings")
        cfg.inputs = _expand_globs(patterns, base_dir)

    # --- output ---
    if "output" in raw:
        out = raw["output"]
        if isinstance(out, str):
            # Shorthand: output: "path.mp4"
            cfg.output_path = out
        elif isinstance(out, dict):
            _warn_unknown_keys(set(out.keys()), KNOWN_OUTPUT_KEYS, "output")
            cfg.output_path = out.get("path")
            cfg.output_codec = out.get("codec")
            if "fps" in out:
                cfg.output_fps = float(out["fps"])
            cfg.output_bitrate = out.get("bitrate")
        else:
            raise ValueError("'output' must be a string or mapping")

    # --- detectors ---
    if "detectors" in raw:
        det = raw["detectors"]
        if not isinstance(det, dict):
            raise ValueError("'detectors' must be a mapping")
        if "scene" in det:
            cfg.scene_weight = float(det["scene"])
        if "audio" in det:
            cfg.audio_weight = float(det["audio"])
        if "motion" in det:
            cfg.motion_weight = float(det["motion"])
        if "semantic" in det:
            cfg.semantic_weight = float(det["semantic"])
        if "emotion" in det:
            cfg.emotion_weight = float(det["emotion"])

    # --- semantic ---
    if "semantic" in raw:
        sem = raw["semantic"]
        if not isinstance(sem, dict):
            raise ValueError("'semantic' must be a mapping")
        _warn_unknown_keys(set(sem.keys()), KNOWN_SEMANTIC_KEYS, "semantic")
        if "prompts" in sem:
            cfg.semantic_prompts = list(sem["prompts"])
        if "negative_prompts" in sem:
            cfg.semantic_negative_prompts = list(sem["negative_prompts"])

    # --- segments ---
    if "segments" in raw:
        seg = raw["segments"]
        if not isinstance(seg, dict):
            raise ValueError("'segments' must be a mapping")
        _warn_unknown_keys(set(seg.keys()), KNOWN_SEGMENT_KEYS, "segments")
        if "threshold" in seg:
            cfg.score_threshold = float(seg["threshold"])
        if "min_duration" in seg:
            cfg.min_segment_duration = float(seg["min_duration"])
        if "max_duration" in seg:
            cfg.max_segment_duration = float(seg["max_duration"])
        if "pre_padding" in seg:
            cfg.pre_padding = float(seg["pre_padding"])
        if "post_padding" in seg:
            cfg.post_padding = float(seg["post_padding"])

    # --- workers ---
    if "workers" in raw:
        cfg.workers = int(raw["workers"])

    return cfg


def apply_config_to_settings(config: PipelineConfig, settings: Settings | None = None) -> Settings:
    """Overlay non-None PipelineConfig fields onto a Settings instance."""
    if settings is None:
        settings = Settings()

    field_map = {
        "scene_weight": "scene_weight",
        "audio_weight": "audio_weight",
        "motion_weight": "motion_weight",
        "semantic_weight": "semantic_weight",
        "emotion_weight": "emotion_weight",
        "semantic_prompts": "semantic_prompts",
        "semantic_negative_prompts": "semantic_negative_prompts",
        "score_threshold": "score_threshold",
        "min_segment_duration": "min_segment_duration_sec",
        "max_segment_duration": "max_segment_duration_sec",
        "pre_padding": "pre_padding_sec",
        "post_padding": "post_padding_sec",
        "workers": "max_workers",
        "output_codec": "output_codec",
        "output_fps": "output_fps",
        "output_bitrate": "output_bitrate",
    }

    for config_field, settings_field in field_map.items():
        value = getattr(config, config_field)
        if value is not None:
            setattr(settings, settings_field, value)

    return settings
