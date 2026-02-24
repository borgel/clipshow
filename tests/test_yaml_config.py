"""Tests for YAML pipeline configuration."""

import os
import textwrap
import warnings

import pytest
import yaml

from clipshow.config import Settings
from clipshow.yaml_config import (
    PipelineConfig,
    apply_config_to_settings,
    load_pipeline_config,
)

# ---------------------------------------------------------------------------
# Parsing tests
# ---------------------------------------------------------------------------

class TestLoadPipelineConfig:
    def test_full_config(self, tmp_path):
        cfg_file = tmp_path / "pipeline.yaml"
        # Create some input files for glob expansion
        (tmp_path / "clip1.mp4").touch()
        (tmp_path / "clip2.mp4").touch()
        cfg_file.write_text(textwrap.dedent("""\
            inputs:
              - "*.mp4"

            output:
              path: "highlight_reel.mp4"
              codec: libx265
              fps: 24.0
              bitrate: "12M"

            detectors:
              scene: 0.3
              audio: 0.25
              motion: 0.25
              semantic: 0.1
              emotion: 0.1

            semantic:
              prompts:
                - "exciting moment"
              negative_prompts:
                - "boring"

            segments:
              threshold: 0.6
              min_duration: 2.0
              max_duration: 10.0
              pre_padding: 0.5
              post_padding: 1.0

            workers: 4
        """))
        cfg = load_pipeline_config(cfg_file)

        assert len(cfg.inputs) == 2
        assert cfg.output_path == "highlight_reel.mp4"
        assert cfg.output_codec == "libx265"
        assert cfg.output_fps == 24.0
        assert cfg.output_bitrate == "12M"
        assert cfg.scene_weight == 0.3
        assert cfg.audio_weight == 0.25
        assert cfg.motion_weight == 0.25
        assert cfg.semantic_weight == 0.1
        assert cfg.emotion_weight == 0.1
        assert cfg.semantic_prompts == ["exciting moment"]
        assert cfg.semantic_negative_prompts == ["boring"]
        assert cfg.score_threshold == 0.6
        assert cfg.min_segment_duration == 2.0
        assert cfg.max_segment_duration == 10.0
        assert cfg.pre_padding == 0.5
        assert cfg.post_padding == 1.0
        assert cfg.workers == 4

    def test_minimal_config(self, tmp_path):
        cfg_file = tmp_path / "minimal.yaml"
        cfg_file.write_text("workers: 2\n")
        cfg = load_pipeline_config(cfg_file)

        assert cfg.inputs == []
        assert cfg.output_path is None
        assert cfg.workers == 2

    def test_empty_file(self, tmp_path):
        cfg_file = tmp_path / "empty.yaml"
        cfg_file.write_text("")
        cfg = load_pipeline_config(cfg_file)

        assert cfg.inputs == []
        assert cfg.workers is None

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_pipeline_config(tmp_path / "nonexistent.yaml")

    def test_invalid_yaml(self, tmp_path):
        cfg_file = tmp_path / "bad.yaml"
        cfg_file.write_text(":\n  :\n    - [invalid")
        with pytest.raises(yaml.YAMLError):
            load_pipeline_config(cfg_file)

    def test_invalid_structure(self, tmp_path):
        cfg_file = tmp_path / "list.yaml"
        cfg_file.write_text("- item1\n- item2\n")
        with pytest.raises(ValueError, match="YAML mapping"):
            load_pipeline_config(cfg_file)

    def test_output_as_string_shorthand(self, tmp_path):
        cfg_file = tmp_path / "short.yaml"
        cfg_file.write_text('output: "my_reel.mp4"\n')
        cfg = load_pipeline_config(cfg_file)

        assert cfg.output_path == "my_reel.mp4"
        assert cfg.output_codec is None

    def test_unknown_keys_warning(self, tmp_path):
        cfg_file = tmp_path / "unknown.yaml"
        cfg_file.write_text(textwrap.dedent("""\
            workers: 2
            foobar: true
            output:
              path: "out.mp4"
              quality: high
        """))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cfg = load_pipeline_config(cfg_file)
            warning_messages = [str(x.message) for x in w]

        assert cfg.workers == 2
        assert any("foobar" in m for m in warning_messages)
        assert any("quality" in m for m in warning_messages)

    def test_inputs_as_single_string(self, tmp_path):
        (tmp_path / "video.mp4").touch()
        cfg_file = tmp_path / "single.yaml"
        cfg_file.write_text('inputs: "video.mp4"\n')
        cfg = load_pipeline_config(cfg_file)

        assert len(cfg.inputs) == 1


# ---------------------------------------------------------------------------
# Glob expansion tests
# ---------------------------------------------------------------------------

class TestGlobExpansion:
    def test_relative_paths(self, tmp_path):
        subdir = tmp_path / "clips"
        subdir.mkdir()
        (subdir / "a.mp4").touch()
        (subdir / "b.mp4").touch()
        (subdir / "c.mov").touch()

        cfg_file = tmp_path / "pipeline.yaml"
        cfg_file.write_text('inputs:\n  - "clips/*.mp4"\n')
        cfg = load_pipeline_config(cfg_file)

        assert len(cfg.inputs) == 2
        assert all(p.endswith(".mp4") for p in cfg.inputs)
        assert all(os.path.isabs(p) for p in cfg.inputs)

    def test_absolute_paths(self, tmp_path):
        (tmp_path / "video.mp4").touch()
        cfg_file = tmp_path / "pipeline.yaml"
        abs_pattern = str(tmp_path / "*.mp4")
        cfg_file.write_text(f"inputs:\n  - '{abs_pattern}'\n")
        cfg = load_pipeline_config(cfg_file)

        assert len(cfg.inputs) == 1

    def test_no_matches(self, tmp_path):
        cfg_file = tmp_path / "pipeline.yaml"
        cfg_file.write_text('inputs:\n  - "*.nonexistent"\n')
        cfg = load_pipeline_config(cfg_file)

        assert cfg.inputs == []

    def test_deduplication(self, tmp_path):
        (tmp_path / "video.mp4").touch()
        cfg_file = tmp_path / "pipeline.yaml"
        cfg_file.write_text('inputs:\n  - "video.mp4"\n  - "*.mp4"\n')
        cfg = load_pipeline_config(cfg_file)

        assert len(cfg.inputs) == 1


# ---------------------------------------------------------------------------
# apply_config_to_settings tests
# ---------------------------------------------------------------------------

class TestApplyConfigToSettings:
    def test_full_overlay(self):
        cfg = PipelineConfig(
            scene_weight=0.5,
            audio_weight=0.3,
            motion_weight=0.1,
            semantic_weight=0.05,
            emotion_weight=0.05,
            semantic_prompts=["custom prompt"],
            semantic_negative_prompts=["bad"],
            score_threshold=0.7,
            min_segment_duration=2.0,
            max_segment_duration=20.0,
            pre_padding=0.5,
            post_padding=2.0,
            workers=8,
            output_codec="libx265",
            output_fps=60.0,
            output_bitrate="16M",
        )
        settings = apply_config_to_settings(cfg)

        assert settings.scene_weight == 0.5
        assert settings.audio_weight == 0.3
        assert settings.motion_weight == 0.1
        assert settings.semantic_weight == 0.05
        assert settings.emotion_weight == 0.05
        assert settings.semantic_prompts == ["custom prompt"]
        assert settings.semantic_negative_prompts == ["bad"]
        assert settings.score_threshold == 0.7
        assert settings.min_segment_duration_sec == 2.0
        assert settings.max_segment_duration_sec == 20.0
        assert settings.pre_padding_sec == 0.5
        assert settings.post_padding_sec == 2.0
        assert settings.max_workers == 8
        assert settings.output_codec == "libx265"
        assert settings.output_fps == 60.0
        assert settings.output_bitrate == "16M"

    def test_partial_overlay(self):
        cfg = PipelineConfig(scene_weight=0.8)
        settings = apply_config_to_settings(cfg)

        assert settings.scene_weight == 0.8
        # Defaults preserved
        assert settings.audio_weight == 0.0
        assert settings.output_codec == "libx264"

    def test_empty_config(self):
        cfg = PipelineConfig()
        defaults = Settings()
        settings = apply_config_to_settings(cfg)

        assert settings.scene_weight == defaults.scene_weight
        assert settings.output_fps == defaults.output_fps

    def test_overlay_onto_existing_settings(self):
        existing = Settings(scene_weight=0.9, audio_weight=0.8)
        cfg = PipelineConfig(scene_weight=0.1)
        settings = apply_config_to_settings(cfg, existing)

        assert settings.scene_weight == 0.1
        assert settings.audio_weight == 0.8  # Untouched


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------

class TestCLIIntegration:
    def test_config_arg_parsed(self):
        from clipshow.__main__ import parse_args

        args = parse_args(["--auto", "--config", "pipeline.yaml"])
        assert args.config == "pipeline.yaml"

    def test_config_short_flag(self):
        from clipshow.__main__ import parse_args

        args = parse_args(["--auto", "-c", "pipeline.yaml"])
        assert args.config == "pipeline.yaml"

    def test_cli_args_override_yaml_output(self, tmp_path):
        cfg_file = tmp_path / "pipeline.yaml"
        cfg_file.write_text('output: "yaml_output.mp4"\n')
        cfg = load_pipeline_config(cfg_file)

        # Simulate CLI precedence: --output was explicitly given
        cli_output = "cli_output.mp4"
        resolved = cli_output or cfg.output_path or "highlight_reel.mp4"
        assert resolved == "cli_output.mp4"

    def test_yaml_output_used_when_cli_omitted(self, tmp_path):
        cfg_file = tmp_path / "pipeline.yaml"
        cfg_file.write_text('output: "yaml_output.mp4"\n')
        cfg = load_pipeline_config(cfg_file)

        # Simulate CLI precedence: --output was not given (None)
        cli_output = None
        resolved = cli_output or cfg.output_path or "highlight_reel.mp4"
        assert resolved == "yaml_output.mp4"

    def test_default_output_when_both_omitted(self):
        cfg = PipelineConfig()
        cli_output = None
        resolved = cli_output or cfg.output_path or "highlight_reel.mp4"
        assert resolved == "highlight_reel.mp4"

    def test_positional_files_override_yaml_inputs(self, tmp_path):
        (tmp_path / "yaml_video.mp4").touch()
        cfg_file = tmp_path / "pipeline.yaml"
        cfg_file.write_text('inputs:\n  - "yaml_video.mp4"\n')
        cfg = load_pipeline_config(cfg_file)

        cli_files = ["cli_video.mp4"]
        resolved = cli_files if cli_files else cfg.inputs
        assert resolved == ["cli_video.mp4"]

    def test_yaml_inputs_used_when_no_positional_args(self, tmp_path):
        (tmp_path / "yaml_video.mp4").touch()
        cfg_file = tmp_path / "pipeline.yaml"
        cfg_file.write_text('inputs:\n  - "yaml_video.mp4"\n')
        cfg = load_pipeline_config(cfg_file)

        cli_files = []
        resolved = cli_files if cli_files else cfg.inputs
        assert len(resolved) == 1

    def test_output_default_is_none(self):
        from clipshow.__main__ import parse_args

        args = parse_args(["--auto"])
        assert args.output is None

    def test_workers_default_is_none(self):
        from clipshow.__main__ import parse_args

        args = parse_args(["--auto"])
        assert args.workers is None
