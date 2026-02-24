"""Audio peak detection via librosa."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

from clipshow.detection.base import Detector, DetectorResult


class AudioDetector(Detector):
    """Detects audio peaks using librosa onset strength and RMS energy.

    Extracts audio to temp WAV via FFmpeg, then runs librosa analysis.
    """

    name = "audio"

    def __init__(self, time_step: float = 0.1, sr: int = 22050):
        self._time_step = time_step
        self._sr = sr

    def _extract_audio(self, video_path: str) -> str | None:
        """Extract audio to a temporary WAV file. Returns path or None if no audio."""
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", video_path,
                    "-vn",
                    "-acodec", "pcm_s16le",
                    "-ar", str(self._sr),
                    "-ac", "1",
                    "-threads", "0",
                    tmp.name,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                Path(tmp.name).unlink(missing_ok=True)
                return None
            # Check file has content
            if Path(tmp.name).stat().st_size < 100:
                Path(tmp.name).unlink(missing_ok=True)
                return None
            return tmp.name
        except (FileNotFoundError, subprocess.TimeoutExpired):
            Path(tmp.name).unlink(missing_ok=True)
            return None

    def detect(
        self,
        video_path: str,
        progress_callback: callable | None = None,
        cancel_flag: callable | None = None,
    ) -> DetectorResult:
        import librosa

        wav_path = self._extract_audio(video_path)
        if wav_path is None:
            # No audio track — return empty scores
            return DetectorResult(
                name=self.name,
                scores=np.array([]),
                time_step=self._time_step,
                source_path=video_path,
            )

        try:
            y, sr = librosa.load(wav_path, sr=self._sr, mono=True)
        finally:
            Path(wav_path).unlink(missing_ok=True)

        if len(y) == 0:
            return DetectorResult(
                name=self.name,
                scores=np.array([]),
                time_step=self._time_step,
                source_path=video_path,
            )

        duration = len(y) / sr
        num_samples = max(1, int(np.ceil(duration / self._time_step)))

        # Onset strength (captures transients, beats, speech onsets)
        hop_length = int(sr * self._time_step)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

        # RMS energy
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

        # Combine: equal weight onset + rms
        min_len = min(len(onset_env), len(rms), num_samples)
        combined = np.zeros(num_samples, dtype=float)
        combined[:min_len] = (
            onset_env[:min_len] / max(onset_env.max(), 1e-10)
            + rms[:min_len] / max(rms.max(), 1e-10)
        ) / 2.0

        # Normalize to [0, 1]
        max_val = combined.max()
        if max_val > 0:
            combined = combined / max_val

        if progress_callback:
            progress_callback(1.0)

        return DetectorResult(
            name=self.name,
            scores=combined,
            time_step=self._time_step,
            source_path=video_path,
        )
