"""LanguageBind-based audio-visual-text detector.

Jointly scores video frames and audio against text prompts using
LanguageBind's shared embedding space. Unlike SemanticDetector (which
only sees video) and AudioDetector (which only sees audio), this detector
understands cross-modal semantics: a cheering crowd sounds exciting AND
looks exciting, and the model was trained to fuse those signals.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from scipy.ndimage import uniform_filter1d

from clipshow.detection.audio_preprocess import (  # noqa: F401
    extract_mel_windows,
    format_for_languagebind,
)
from clipshow.detection.base import Detector, DetectorResult
from clipshow.detection.models import ModelManager  # noqa: F401

logger = logging.getLogger(__name__)

DEFAULT_PROMPTS = [
    "exciting moment",
    "people laughing",
    "beautiful scenery",
    "action scene",
]
DEFAULT_NEGATIVE_PROMPTS = [
    "boring static shot",
    "blank wall",
    "empty room",
    "black screen",
]

# Video frame sampling rate
SAMPLE_FPS = 2

# Sigmoid normalization parameters
_SIGMOID_CENTER = 0.0
_SIGMOID_SCALE = 5.0

# Temporal smoothing window (in score samples)
_SMOOTH_WINDOW = 5

# CLIP-style image normalization stats
_IMAGE_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_IMAGE_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


class AudioVisualDetector(Detector):
    """LanguageBind-based audio-visual-text detector."""

    name = "audiovisual"

    def __init__(
        self,
        prompts: list[str] | None = None,
        negative_prompts: list[str] | None = None,
        audio_weight: float = 0.4,
        time_step: float = 0.1,
    ):
        self._prompts = prompts or DEFAULT_PROMPTS
        self._negative_prompts = negative_prompts or DEFAULT_NEGATIVE_PROMPTS
        self._audio_weight = audio_weight
        self._time_step = time_step
        self._video_session = None
        self._audio_session = None
        self._text_session = None

    def _load_models(self, progress_callback=None):
        """Load all three LanguageBind ONNX sessions."""
        manager = ModelManager()
        self._video_session = manager.load_session("languagebind-video")
        self._audio_session = manager.load_session("languagebind-audio")
        self._text_session = manager.load_session("languagebind-text")

    def _preprocess_frames(self, frames: list[np.ndarray]) -> np.ndarray:
        """Preprocess sampled BGR frames for LanguageBind video encoder.

        Resizes to 224x224, converts BGR->RGB, normalizes with CLIP stats.
        Returns array of shape (N, 3, 224, 224).
        """
        processed = []
        for frame in frames:
            resized = cv2.resize(frame, (224, 224))
            rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
            rgb = (rgb - _IMAGE_MEAN) / _IMAGE_STD
            chw = np.transpose(rgb, (2, 0, 1))
            processed.append(chw)
        return np.stack(processed).astype(np.float32)

    def _tokenize_prompts(self, prompts: list[str]) -> dict[str, np.ndarray]:
        """Simple text tokenization for LanguageBind text encoder.

        Creates input_ids and attention_mask. Uses character-level encoding
        as a placeholder; production use should load the saved BPE tokenizer.
        """
        max_len = 77
        batch_ids = []
        batch_mask = []
        for text in prompts:
            ids = [49406]  # BOS
            for ch in text[: max_len - 2]:
                ids.append(ord(ch) % 49406)
            ids.append(49407)  # EOS
            mask = [1] * len(ids) + [0] * (max_len - len(ids))
            ids = ids + [0] * (max_len - len(ids))
            batch_ids.append(ids[:max_len])
            batch_mask.append(mask[:max_len])
        return {
            "input_ids": np.array(batch_ids, dtype=np.int64),
            "attention_mask": np.array(batch_mask, dtype=np.int64),
        }

    def detect(
        self,
        video_path: str,
        progress_callback: callable | None = None,
        cancel_flag: callable | None = None,
    ) -> DetectorResult:
        if self._video_session is None:
            self._load_models(progress_callback)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0.0
        frame_interval = max(1, int(fps / SAMPLE_FPS))
        num_samples = max(1, int(np.ceil(duration / self._time_step)))

        # 1. Sample video frames
        frames = []
        frame_times = []
        frame_idx = 0
        while True:
            if cancel_flag and cancel_flag():
                break
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                frames.append(frame)
                frame_times.append(frame_idx / fps)
                if progress_callback:
                    progress_callback(0.3 * frame_idx / max(total_frames, 1))
            frame_idx += 1
        cap.release()

        if not frames or (cancel_flag and cancel_flag()):
            scores = np.zeros(num_samples, dtype=float)
            if progress_callback:
                progress_callback(1.0)
            return DetectorResult(
                name=self.name,
                scores=scores,
                time_step=self._time_step,
                source_path=video_path,
            )

        frame_times_arr = np.array(frame_times)

        # 2. Encode video frames
        video_input = self._preprocess_frames(frames)
        video_emb = self._video_session.run(
            None, {"pixel_values": video_input}
        )[0]

        if progress_callback:
            progress_callback(0.5)

        # 3. Extract and encode audio
        mel_windows = extract_mel_windows(video_path, frame_times_arr)
        has_audio = mel_windows is not None
        audio_emb = None
        if has_audio:
            audio_input = format_for_languagebind(mel_windows)
            audio_emb = self._audio_session.run(
                None, {"pixel_values": audio_input}
            )[0]

        if progress_callback:
            progress_callback(0.7)

        # 4. Encode text prompts
        text_pos = self._text_session.run(
            None, self._tokenize_prompts(self._prompts)
        )[0]
        text_neg = self._text_session.run(
            None, self._tokenize_prompts(self._negative_prompts)
        )[0]

        if progress_callback:
            progress_callback(0.8)

        # 5. Fuse audio + video similarities
        video_pos = video_emb @ text_pos.T
        video_neg = video_emb @ text_neg.T

        alpha = 1.0 - self._audio_weight

        if has_audio and audio_emb is not None:
            audio_pos = audio_emb @ text_pos.T
            audio_neg = audio_emb @ text_neg.T

            # Align audio to video frame count if lengths differ
            n_video = len(video_pos)
            n_audio = len(audio_pos)
            if n_audio != n_video:
                x_audio = np.linspace(0, 1, n_audio)
                x_video = np.linspace(0, 1, n_video)
                audio_pos = np.column_stack(
                    [
                        np.interp(x_video, x_audio, audio_pos[:, j])
                        for j in range(audio_pos.shape[1])
                    ]
                )
                audio_neg = np.column_stack(
                    [
                        np.interp(x_video, x_audio, audio_neg[:, j])
                        for j in range(audio_neg.shape[1])
                    ]
                )

            fused_pos = alpha * video_pos + (1 - alpha) * audio_pos
            fused_neg = alpha * video_neg + (1 - alpha) * audio_neg
        else:
            fused_pos = video_pos
            fused_neg = video_neg

        raw = fused_pos.max(axis=1) - fused_neg.max(axis=1)

        # 6. Sigmoid normalization
        frame_scores = 1.0 / (
            1.0 + np.exp(-_SIGMOID_SCALE * (raw - _SIGMOID_CENTER))
        )
        frame_scores = np.clip(frame_scores, 0.0, 1.0)

        # 7. Map frame scores to time-step grid
        scores = np.zeros(num_samples, dtype=float)
        for i, t in enumerate(frame_times):
            if i < len(frame_scores):
                idx = min(int(t / self._time_step), num_samples - 1)
                scores[idx] = max(scores[idx], float(frame_scores[i]))

        # 8. Temporal smoothing
        if len(scores) >= _SMOOTH_WINDOW:
            scores = uniform_filter1d(scores, size=_SMOOTH_WINDOW)

        # 9. Final normalization to [0, 1]
        max_val = scores.max()
        if max_val > 0:
            scores = scores / max_val

        if progress_callback:
            progress_callback(1.0)

        return DetectorResult(
            name=self.name,
            scores=scores,
            time_step=self._time_step,
            source_path=video_path,
        )
