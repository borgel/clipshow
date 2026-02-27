"""Video metadata extraction via ffprobe."""

import json
import subprocess
from pathlib import Path

from clipshow.model.project import VideoSource


def probe(video_path: str) -> dict:
    """Run ffprobe and return parsed JSON output for the first video stream.

    Raises:
        FileNotFoundError: If video_path does not exist.
        RuntimeError: If ffprobe fails or no video stream found.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "ffprobe not found. Install FFmpeg: brew install ffmpeg (macOS) "
            "or sudo apt install ffmpeg (Linux)"
        )

    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr.strip()}")

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"ffprobe returned invalid JSON: {e}")

    return data


def get_video_stream(data: dict) -> dict:
    """Extract the first video stream from ffprobe data.

    Raises:
        RuntimeError: If no video stream found.
    """
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            return stream
    raise RuntimeError("No video stream found in file")


def extract_metadata(video_path: str) -> VideoSource:
    """Extract video metadata and return a populated VideoSource."""
    data = probe(video_path)
    stream = get_video_stream(data)
    fmt = data.get("format", {})

    # Parse FPS from r_frame_rate (e.g., "24/1" or "30000/1001")
    fps = 0.0
    r_frame_rate = stream.get("r_frame_rate", "0/1")
    try:
        num, den = r_frame_rate.split("/")
        if int(den) != 0:
            fps = int(num) / int(den)
    except (ValueError, ZeroDivisionError):
        pass

    # Duration: try multiple sources — some containers omit format/stream duration
    duration = 0.0
    for source in [fmt, stream]:
        raw = source.get("duration")
        if raw is not None:
            try:
                duration = float(raw)
                if duration > 0:
                    break
            except (ValueError, TypeError):
                pass

    # Fallback: compute from nb_frames / fps
    if duration <= 0 and fps > 0:
        nb_frames = stream.get("nb_frames")
        if nb_frames is not None:
            try:
                duration = int(nb_frames) / fps
            except (ValueError, TypeError):
                pass

    # Fallback: tags.DURATION (common in MKV/MTS containers, format "HH:MM:SS.microseconds")
    if duration <= 0:
        for source in [stream, fmt]:
            tag_dur = source.get("tags", {}).get("DURATION")
            if tag_dur:
                try:
                    parts = tag_dur.split(":")
                    duration = (
                        float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
                    )
                    if duration > 0:
                        break
                except (ValueError, IndexError):
                    pass

    return VideoSource(
        path=video_path,
        duration=duration,
        width=int(stream.get("width", 0)),
        height=int(stream.get("height", 0)),
        fps=round(fps, 3),
        codec=stream.get("codec_name", ""),
    )
