"""Shared test fixtures: synthetic videos, QApplication, offscreen setup."""

import os
import tempfile

import numpy as np
import pytest

# Force offscreen rendering for headless CI
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def pytest_addoption(parser):
    parser.addoption(
        "--update-baselines",
        action="store_true",
        default=False,
        help="Regenerate visual regression baseline images.",
    )


@pytest.fixture(scope="session")
def tmp_video_dir():
    """Session-scoped temp directory for synthetic test videos."""
    with tempfile.TemporaryDirectory(prefix="clipshow_test_") as d:
        yield d


def _make_video(path: str, duration: float, fps: float, size: tuple[int, int], make_frame):
    """Helper to create a synthetic video using MoviePy."""
    from moviepy import VideoClip

    clip = VideoClip(make_frame, duration=duration).with_fps(fps)
    clip = clip.resized(size)
    clip.write_videofile(
        path,
        codec="libx264",
        audio=False,
        logger=None,
    )
    clip.close()


def _make_video_with_audio(
    path: str,
    duration: float,
    fps: float,
    size: tuple[int, int],
    make_frame,
    make_audio,
    audio_fps: int = 44100,
):
    """Helper to create a synthetic video with audio using MoviePy."""
    from moviepy import AudioClip, VideoClip

    video = VideoClip(make_frame, duration=duration).with_fps(fps)
    video = video.resized(size)
    audio = AudioClip(make_audio, duration=duration, fps=audio_fps)
    video = video.with_audio(audio)
    video.write_videofile(
        path,
        codec="libx264",
        audio_codec="aac",
        logger=None,
    )
    video.close()


@pytest.fixture(scope="session")
def static_video(tmp_video_dir):
    """Uniform blue video — no motion, no scene changes. 2s @ 24fps, 160x120."""
    path = os.path.join(tmp_video_dir, "static.mp4")

    def make_frame(t):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        frame[:, :] = [40, 60, 200]  # Blue
        return frame

    _make_video(path, duration=2.0, fps=24, size=(160, 120), make_frame=make_frame)
    return path


@pytest.fixture(scope="session")
def scene_change_video(tmp_video_dir):
    """Video with abrupt color change at 50% mark. 2s @ 24fps, 160x120."""
    path = os.path.join(tmp_video_dir, "scene_change.mp4")

    def make_frame(t):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        if t < 1.0:
            frame[:, :] = [200, 40, 40]  # Red first half
        else:
            frame[:, :] = [40, 200, 40]  # Green second half
        return frame

    _make_video(path, duration=2.0, fps=24, size=(160, 120), make_frame=make_frame)
    return path


@pytest.fixture(scope="session")
def motion_video(tmp_video_dir):
    """Video with a bright moving object on dark background. 2s @ 24fps, 160x120."""
    path = os.path.join(tmp_video_dir, "motion.mp4")

    def make_frame(t):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        # Moving white circle across the frame
        cx = int((t / 2.0) * 160)
        cy = 60
        for y in range(max(0, cy - 15), min(120, cy + 15)):
            for x in range(max(0, cx - 15), min(160, cx + 15)):
                if (x - cx) ** 2 + (y - cy) ** 2 < 15**2:
                    frame[y, x] = [255, 255, 255]
        return frame

    _make_video(path, duration=2.0, fps=24, size=(160, 120), make_frame=make_frame)
    return path


@pytest.fixture(scope="session")
def loud_moment_video(tmp_video_dir):
    """Video with silence then burst of white noise audio at ~1.0s. 2s @ 24fps, 160x120."""
    path = os.path.join(tmp_video_dir, "loud_moment.mp4")

    def make_frame(t):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        frame[:, :] = [100, 100, 100]  # Gray
        return frame

    def make_audio(t):
        # t can be a scalar (float/int) or array of time values
        t = np.atleast_1d(np.asarray(t, dtype=float))
        rng = np.random.default_rng(42)
        audio = np.where(
            (t >= 1.0) & (t < 1.5),
            rng.standard_normal(t.shape) * 0.8,
            np.zeros_like(t) + 0.001,  # Near-silence
        )
        # Return as (N, 1) for mono
        if audio.ndim == 1:
            audio = audio[:, np.newaxis]
        return audio

    _make_video_with_audio(
        path,
        duration=2.0,
        fps=24,
        size=(160, 120),
        make_frame=make_frame,
        make_audio=make_audio,
    )
    return path
