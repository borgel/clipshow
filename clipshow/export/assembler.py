"""Video clip concatenation and encoding via MoviePy."""

from __future__ import annotations

from moviepy import VideoFileClip, concatenate_videoclips

from clipshow.model.moments import HighlightSegment


def assemble_highlights(
    segments: list[HighlightSegment],
    output_path: str,
    codec: str = "libx264",
    fps: float = 30.0,
    bitrate: str = "8M",
    progress_callback: callable | None = None,
) -> str:
    """Concatenate video segments into a single highlight reel.

    Args:
        segments: Ordered list of HighlightSegments (only included ones).
        output_path: Path for the output MP4 file.
        codec: Video codec (default: libx264).
        fps: Output framerate.
        bitrate: Output bitrate string.
        progress_callback: Optional callable(float) for progress (0-1).

    Returns:
        The output_path on success.

    Raises:
        ValueError: If no segments provided.
        RuntimeError: If assembly fails.
    """
    included = [s for s in segments if s.included]
    if not included:
        raise ValueError("No included segments to assemble")

    clips = []
    try:
        for i, seg in enumerate(included):
            clip = VideoFileClip(seg.source_path)
            # Clamp subclip to valid range
            start = max(0, seg.start_time)
            end = min(clip.duration, seg.end_time) if clip.duration else seg.end_time
            subclip = clip.subclipped(start, end)
            clips.append(subclip)

            if progress_callback:
                progress_callback((i + 1) / (len(included) + 1))

        final = concatenate_videoclips(clips, method="compose")
        final.write_videofile(
            output_path,
            codec=codec,
            fps=fps,
            bitrate=bitrate,
            logger=None,
        )

        if progress_callback:
            progress_callback(1.0)

        return output_path
    finally:
        for clip in clips:
            clip.close()
