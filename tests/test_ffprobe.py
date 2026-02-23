"""Tests for ffprobe metadata extraction."""

import pytest

from clipshow.export.ffprobe import extract_metadata, get_video_stream, probe


class TestProbe:
    def test_probe_returns_dict(self, static_video):
        data = probe(static_video)
        assert isinstance(data, dict)
        assert "streams" in data
        assert "format" in data

    def test_probe_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            probe(str(tmp_path / "nonexistent.mp4"))

    def test_probe_invalid_file(self, tmp_path):
        bad_file = tmp_path / "not_a_video.txt"
        bad_file.write_text("this is not a video")
        with pytest.raises(RuntimeError):
            probe(str(bad_file))


class TestGetVideoStream:
    def test_extracts_video_stream(self, static_video):
        data = probe(static_video)
        stream = get_video_stream(data)
        assert stream["codec_type"] == "video"

    def test_no_video_stream_raises(self):
        with pytest.raises(RuntimeError, match="No video stream"):
            get_video_stream({"streams": [{"codec_type": "audio"}]})

    def test_empty_streams_raises(self):
        with pytest.raises(RuntimeError, match="No video stream"):
            get_video_stream({"streams": []})


class TestExtractMetadata:
    def test_static_video_metadata(self, static_video):
        source = extract_metadata(static_video)
        assert source.path == static_video
        assert source.width == 160
        assert source.height == 120
        assert source.fps == pytest.approx(24.0, abs=0.1)
        assert source.duration == pytest.approx(2.0, abs=0.2)
        assert source.codec == "h264"

    def test_scene_change_video_metadata(self, scene_change_video):
        source = extract_metadata(scene_change_video)
        assert source.width == 160
        assert source.height == 120
        assert source.duration == pytest.approx(2.0, abs=0.2)

    def test_motion_video_metadata(self, motion_video):
        source = extract_metadata(motion_video)
        assert source.width == 160
        assert source.height == 120

    def test_loud_moment_video_metadata(self, loud_moment_video):
        source = extract_metadata(loud_moment_video)
        assert source.width == 160
        assert source.height == 120
        # This video has audio
        assert source.duration == pytest.approx(2.0, abs=0.2)

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            extract_metadata(str(tmp_path / "nonexistent.mp4"))
