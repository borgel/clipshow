"""Tests for audio preprocessing (LanguageBind mel-spectrogram extraction)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from clipshow.detection.audio_preprocess import (
    AUDIO_MEAN,
    AUDIO_SR,
    AUDIO_STD,
    N_MELS,
    TARGET_LENGTH,
    compute_mel_spectrogram,
    extract_audio_wav,
    extract_mel_windows,
    format_for_languagebind,
)

# ---------------------------------------------------------------------------
# extract_audio_wav
# ---------------------------------------------------------------------------


class TestExtractAudioWav:
    def test_returns_none_when_no_audio(self):
        """Should return None for a video with no audio track."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            result = extract_audio_wav("/fake/video.mp4")
        assert result is None

    def test_returns_none_when_ffmpeg_missing(self):
        """Should return None if ffmpeg is not found."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = extract_audio_wav("/fake/video.mp4")
        assert result is None

    def test_returns_none_for_empty_output(self, tmp_path):
        """Should return None if extracted WAV is too small."""
        # Create a tiny file that will be "too small"
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            with patch("pathlib.Path.stat") as mock_stat:
                mock_stat.return_value = MagicMock(st_size=10)
                result = extract_audio_wav("/fake/video.mp4")
        assert result is None

    def test_uses_correct_sample_rate(self):
        """Should pass the sample rate to ffmpeg."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            extract_audio_wav("/fake/video.mp4", sr=16000)

        call_args = mock_run.call_args[0][0]
        ar_idx = call_args.index("-ar")
        assert call_args[ar_idx + 1] == "16000"

    def test_extracts_mono(self):
        """Should extract mono audio (-ac 1)."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            extract_audio_wav("/fake/video.mp4")

        call_args = mock_run.call_args[0][0]
        ac_idx = call_args.index("-ac")
        assert call_args[ac_idx + 1] == "1"


# ---------------------------------------------------------------------------
# compute_mel_spectrogram
# ---------------------------------------------------------------------------


class TestComputeMelSpectrogram:
    def test_output_shape(self):
        """Should return (n_mels, time_frames) array."""
        librosa = pytest.importorskip("librosa")  # noqa: F841
        # 2 seconds of silence at 16kHz
        chunk = np.zeros(32000, dtype=np.float32)
        mel = compute_mel_spectrogram(chunk)
        assert mel.ndim == 2
        assert mel.shape[0] == N_MELS

    def test_custom_n_mels(self):
        """Should respect custom n_mels parameter."""
        pytest.importorskip("librosa")
        chunk = np.zeros(32000, dtype=np.float32)
        mel = compute_mel_spectrogram(chunk, n_mels=64)
        assert mel.shape[0] == 64

    def test_nonzero_audio_has_variation(self):
        """Non-silent audio should produce varying mel values."""
        pytest.importorskip("librosa")
        rng = np.random.default_rng(42)
        chunk = rng.standard_normal(32000).astype(np.float32) * 0.5
        mel = compute_mel_spectrogram(chunk)
        # Should not be all the same value
        assert mel.std() > 0


# ---------------------------------------------------------------------------
# extract_mel_windows
# ---------------------------------------------------------------------------


class TestExtractMelWindows:
    def test_returns_none_when_no_audio(self):
        """Should return None when video has no audio track."""
        with patch(
            "clipshow.detection.audio_preprocess.extract_audio_wav",
            return_value=None,
        ):
            result = extract_mel_windows(
                "/fake/video.mp4", np.array([0.5, 1.0, 1.5])
            )
        assert result is None

    def test_output_shape_matches_sample_times(self, tmp_path):
        """Output should have one window per sample time."""
        pytest.importorskip("librosa")

        # Create a fake WAV file
        wav_path = tmp_path / "test.wav"
        wav_path.write_bytes(b"x" * 200)  # dummy

        sample_times = np.array([0.5, 1.0, 1.5, 2.0])

        # Mock extract_audio_wav to return our file, mock librosa.load
        with patch(
            "clipshow.detection.audio_preprocess.extract_audio_wav",
            return_value=str(wav_path),
        ):
            # 4 seconds of audio at 16kHz
            fake_audio = np.zeros(64000, dtype=np.float32)
            with patch("librosa.load", return_value=(fake_audio, AUDIO_SR)):
                result = extract_mel_windows(
                    "/fake/video.mp4", sample_times
                )

        assert result is not None
        assert result.shape[0] == len(sample_times)
        assert result.shape[1] == N_MELS

    def test_pads_short_chunks(self, tmp_path):
        """Chunks near audio boundaries should be zero-padded."""
        pytest.importorskip("librosa")

        wav_path = tmp_path / "test.wav"
        wav_path.write_bytes(b"x" * 200)

        # Very short audio (0.5 seconds) with sample time near the end
        short_audio = np.zeros(8000, dtype=np.float32)
        sample_times = np.array([0.25])

        with patch(
            "clipshow.detection.audio_preprocess.extract_audio_wav",
            return_value=str(wav_path),
        ):
            with patch("librosa.load", return_value=(short_audio, AUDIO_SR)):
                result = extract_mel_windows(
                    "/fake/video.mp4", sample_times
                )

        # Should succeed without error (padding applied)
        assert result is not None
        assert result.shape[0] == 1

    def test_returns_none_for_empty_audio(self, tmp_path):
        """Should return None if audio track is empty."""
        wav_path = tmp_path / "test.wav"
        wav_path.write_bytes(b"x" * 200)

        with patch(
            "clipshow.detection.audio_preprocess.extract_audio_wav",
            return_value=str(wav_path),
        ):
            with patch("librosa.load", return_value=(np.array([]), AUDIO_SR)):
                result = extract_mel_windows(
                    "/fake/video.mp4", np.array([0.5])
                )

        assert result is None


# ---------------------------------------------------------------------------
# format_for_languagebind
# ---------------------------------------------------------------------------


class TestFormatForLanguageBind:
    def test_output_shape(self):
        """Should produce (T, 3, n_mels, target_length) output."""
        # Simulate 3 mel windows with arbitrary freq bins
        mel_windows = np.random.randn(3, N_MELS, 200).astype(np.float32)
        result = format_for_languagebind(mel_windows)
        assert result.shape == (3, 3, N_MELS, TARGET_LENGTH)

    def test_output_dtype_float32(self):
        """Output should be float32 for ONNX Runtime."""
        mel_windows = np.random.randn(2, N_MELS, 100).astype(np.float64)
        result = format_for_languagebind(mel_windows)
        assert result.dtype == np.float32

    def test_pads_short_spectrograms(self):
        """Spectrograms shorter than target_length should be padded."""
        mel_windows = np.ones((1, N_MELS, 50), dtype=np.float32)
        result = format_for_languagebind(mel_windows)
        assert result.shape == (1, 3, N_MELS, TARGET_LENGTH)

    def test_truncates_long_spectrograms(self):
        """Spectrograms longer than target_length should be truncated."""
        mel_windows = np.ones((1, N_MELS, TARGET_LENGTH + 500), dtype=np.float32)
        result = format_for_languagebind(mel_windows)
        assert result.shape == (1, 3, N_MELS, TARGET_LENGTH)

    def test_normalization_applied(self):
        """Output should be normalized with LanguageBind audio stats."""
        # All-zero mel should normalize to -mean/std
        mel_windows = np.zeros((1, N_MELS, TARGET_LENGTH), dtype=np.float32)
        result = format_for_languagebind(mel_windows)
        expected = (0.0 - AUDIO_MEAN) / AUDIO_STD
        # Check first element of first channel
        np.testing.assert_almost_equal(result[0, 0, 0, 0], expected, decimal=4)

    def test_three_channels_are_temporal_crops(self):
        """The 3 channels should be front/mid/back crops of the spectrogram."""
        # Create a spectrogram with a clear gradient so we can verify crops
        mel = np.arange(N_MELS * TARGET_LENGTH, dtype=np.float32).reshape(
            N_MELS, TARGET_LENGTH
        )
        mel_windows = mel[np.newaxis, :, :]  # (1, n_mels, target_length)
        result = format_for_languagebind(mel_windows)

        # Front and back channels should differ (different temporal crops)
        assert not np.allclose(result[0, 0], result[0, 2])

    def test_single_window(self):
        """Should work with a single mel window."""
        mel_windows = np.random.randn(1, N_MELS, 100).astype(np.float32)
        result = format_for_languagebind(mel_windows)
        assert result.shape[0] == 1
