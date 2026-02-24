"""Audio preprocessing for LanguageBind audio-visual detection.

Extracts audio from video via ffmpeg, splits into windowed segments
aligned to video sample times, and computes mel-spectrograms using
librosa. Output feeds into the LanguageBind audio encoder ONNX model.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# LanguageBind audio parameters
AUDIO_SR = 16000  # Sample rate expected by LanguageBind
WINDOW_SEC = 2.0  # Duration of each audio window
N_MELS = 128  # Mel bands (matching LanguageBind training)
N_FFT = 1024
HOP_LENGTH = 320

# LanguageBind audio normalization stats (from model config)
AUDIO_MEAN = -4.2677393
AUDIO_STD = 4.5689974

# LanguageBind target spectrogram length (time steps after fbank)
TARGET_LENGTH = 1036


def extract_audio_wav(video_path: str, sr: int = AUDIO_SR) -> str | None:
    """Extract audio from video to a temporary WAV file.

    Reuses the same ffmpeg pattern as AudioDetector.

    Args:
        video_path: Path to the input video file.
        sr: Target sample rate (default: 16kHz for LanguageBind).

    Returns:
        Path to the temporary WAV file, or None if no audio track.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                video_path,
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                str(sr),
                "-ac",
                "1",
                "-threads",
                "0",
                tmp.name,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            Path(tmp.name).unlink(missing_ok=True)
            return None
        if Path(tmp.name).stat().st_size < 100:
            Path(tmp.name).unlink(missing_ok=True)
            return None
        return tmp.name
    except (FileNotFoundError, subprocess.TimeoutExpired):
        Path(tmp.name).unlink(missing_ok=True)
        return None


def compute_mel_spectrogram(
    audio_chunk: np.ndarray,
    sr: int = AUDIO_SR,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """Compute log-mel spectrogram for a single audio chunk.

    Args:
        audio_chunk: 1-D float32 audio waveform.
        sr: Sample rate.
        n_mels: Number of mel bands.
        n_fft: FFT window size.
        hop_length: Hop length between frames.

    Returns:
        Log-mel spectrogram of shape (n_mels, time_frames).
    """
    import librosa

    mel = librosa.feature.melspectrogram(
        y=audio_chunk,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db


def extract_mel_windows(
    video_path: str,
    sample_times: np.ndarray,
    sr: int = AUDIO_SR,
    window_sec: float = WINDOW_SEC,
) -> np.ndarray | None:
    """Extract mel-spectrogram windows aligned to video sample times.

    Args:
        video_path: Path to the input video file.
        sample_times: 1-D array of center times (seconds) for each window.
        sr: Target sample rate.
        window_sec: Duration of each audio window in seconds.

    Returns:
        Array of shape (T, n_mels, freq_bins) where T = len(sample_times),
        or None if the video has no audio track.
    """
    import librosa

    wav_path = extract_audio_wav(video_path, sr=sr)
    if wav_path is None:
        return None

    try:
        audio, _ = librosa.load(wav_path, sr=sr, mono=True)
    finally:
        Path(wav_path).unlink(missing_ok=True)

    if len(audio) == 0:
        return None

    window_samples = int(window_sec * sr)
    windows = []

    for t in sample_times:
        center = int(t * sr)
        start = max(0, center - window_samples // 2)
        end = start + window_samples
        chunk = audio[start:end]

        # Pad if chunk is shorter than window (near start/end of audio)
        if len(chunk) < window_samples:
            chunk = np.pad(chunk, (0, window_samples - len(chunk)))

        mel = compute_mel_spectrogram(chunk, sr=sr)
        windows.append(mel)

    return np.stack(windows)  # (T, n_mels, freq_bins)


def format_for_languagebind(
    mel_windows: np.ndarray,
    target_length: int = TARGET_LENGTH,
    audio_mean: float = AUDIO_MEAN,
    audio_std: float = AUDIO_STD,
) -> np.ndarray:
    """Format mel windows for LanguageBind audio encoder input.

    LanguageBind expects (B, 3, n_mels, target_length) where the 3 channels
    are front/mid/back temporal crops of the spectrogram.

    Args:
        mel_windows: Array of shape (T, n_mels, freq_bins) from extract_mel_windows.
        target_length: Target time dimension (default: 1036).
        audio_mean: Normalization mean from LanguageBind config.
        audio_std: Normalization std from LanguageBind config.

    Returns:
        Array of shape (T, 3, n_mels, target_length) ready for ONNX inference.
    """
    batch = []
    for mel in mel_windows:
        # mel: (n_mels, freq_bins)
        n_mels_dim, freq_bins = mel.shape

        # Pad or truncate to target_length
        if freq_bins < target_length:
            mel = np.pad(mel, ((0, 0), (0, target_length - freq_bins)))
        elif freq_bins > target_length:
            mel = mel[:, :target_length]

        # Create 3 temporal crops (front, mid, back)
        # This matches LanguageBind's audio processing
        total = mel.shape[1]
        chunk_len = total // 3
        front = mel[:, :chunk_len]
        mid_start = (total - chunk_len) // 2
        mid = mel[:, mid_start : mid_start + chunk_len]
        back = mel[:, total - chunk_len :]

        # Pad each crop back to target_length
        def _pad_to(arr, length):
            if arr.shape[1] < length:
                return np.pad(arr, ((0, 0), (0, length - arr.shape[1])))
            return arr[:, :length]

        stacked = np.stack(
            [
                _pad_to(front, target_length),
                _pad_to(mid, target_length),
                _pad_to(back, target_length),
            ]
        )  # (3, n_mels, target_length)

        # Normalize with LanguageBind stats
        stacked = (stacked - audio_mean) / audio_std
        batch.append(stacked)

    return np.stack(batch).astype(np.float32)  # (T, 3, n_mels, target_length)
