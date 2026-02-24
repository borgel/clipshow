# Semantic Detector Improvement Plan

**Bead**: `clipshow-5us`
**Date**: 2026-02-24
**Status**: Draft

## Current State

The semantic detector (`clipshow/detection/semantic.py`) uses CLIP ViT-B/32 via `onnx_clip` to score video frames against user text prompts. Key characteristics:

- **Model**: CLIP ViT-B/32 (~338MB), loaded via `onnx_clip.OnnxClip(batch_size=1)`
- **Sampling**: 2 FPS (`SAMPLE_FPS = 2`), sequential frame read via OpenCV
- **Inference**: One frame at a time, CPU only (CoreML falls back)
- **Scoring**: Max cosine similarity against positive prompts minus max against negative prompts, sigmoid-normalized
- **Smoothing**: 5-sample uniform filter, then normalize to [0, 1]
- **Speed**: ~60-150s for a 5-minute video on CPU

## Phase 1: Batch Processing

**Impact**: 2-4x speed improvement with minimal code change.
**Difficulty**: Low
**Files**: `semantic.py`

### What Changes

Replace the frame-by-frame inference loop with batched inference. Currently each sampled frame runs through the ONNX model individually. CLIP's image encoder is a standard vision transformer that handles batched inputs efficiently — the per-frame overhead (session setup, memory allocation) dominates at batch_size=1.

### Approach

1. **Accumulate frames in a buffer** instead of calling `get_image_embeddings` per frame.
2. **Flush the buffer** every N frames (default 16) or at end-of-video.
3. **Vectorize similarity scoring** — compute cosine similarity for the whole batch as a single matrix multiply.

```python
BATCH_SIZE = 16

# In detect():
batch_imgs = []
batch_times = []

# Inside frame loop:
batch_imgs.append(img)
batch_times.append(t)
if len(batch_imgs) >= BATCH_SIZE:
    _score_batch(batch_imgs, batch_times, ...)
    batch_imgs.clear()
    batch_times.clear()

# After loop:
if batch_imgs:
    _score_batch(batch_imgs, batch_times, ...)

def _score_batch(self, imgs, times, pos_emb, neg_emb, scores, num_samples):
    image_embeddings = model.get_image_embeddings(imgs)  # (N, 512)
    pos_sims = (image_embeddings @ pos_emb.T).max(axis=1)  # (N,)
    neg_sims = (image_embeddings @ neg_emb.T).max(axis=1)  # (N,)
    raw = pos_sims - neg_sims
    batch_scores = 1.0 / (1.0 + np.exp(-_SIGMOID_SCALE * (raw - _SIGMOID_CENTER)))
    for score, t in zip(batch_scores, times):
        idx = min(int(t / self._time_step), num_samples - 1)
        scores[idx] = max(scores[idx], float(score))
```

The `onnx_clip.OnnxClip` constructor takes `batch_size` — change from 1 to the configured batch size. If `onnx_clip` doesn't support dynamic batching, we fix to BATCH_SIZE at construction.

### Config Changes

Add `semantic_batch_size: int = 16` to `Settings` and the YAML `semantic:` block. Exposed in the settings dialog is not necessary — this is a performance tuning knob, not a user-facing creative control.

### Risks

- `onnx_clip` may not support dynamic batch sizes — need to verify. If not, we can pad the final batch with dummy images and discard those embeddings.
- Memory: 16 frames at 224x224x3 float32 = ~9.6MB. Acceptable.

---

## Phase 2: Prompt Ensemble

**Impact**: 5-15% quality improvement (based on OpenAI's own CLIP evaluation findings).
**Difficulty**: Low
**Files**: `semantic.py`

### What Changes

CLIP was trained on image-caption pairs, not single labels. A prompt like "exciting moment" lands in a different embedding subspace than "a video frame of an exciting moment." OpenAI's recommended practice is to average embeddings across multiple prompt templates for each concept.

### Approach

Define a set of templates and apply each user prompt through all of them before averaging:

```python
PROMPT_TEMPLATES = [
    "a video frame showing {}",
    "a photo of {}",
    "a screenshot of {}",
    "a video clip of {}",
    "{}",  # raw prompt as-is
]

def _build_ensembled_embeddings(self, prompts: list[str]) -> np.ndarray:
    """Create averaged embeddings across prompt templates."""
    all_expanded = []
    for prompt in prompts:
        expanded = [t.format(prompt) for t in PROMPT_TEMPLATES]
        embeddings = model.get_text_embeddings(expanded)  # (T, 512)
        # L2-normalize each, then mean, then re-normalize
        normed = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        mean_emb = normed.mean(axis=0)
        mean_emb /= np.linalg.norm(mean_emb)
        all_expanded.append(mean_emb)
    return np.array(all_expanded)  # (P, 512)
```

This replaces the current single-call `model.get_text_embeddings(self._prompts)`. Text encoding is fast (< 100ms total even with 5x expansion) and happens once at the start — no runtime cost.

### Config Changes

None required. Templates are internal implementation details. Power users could override via a future `semantic.templates` YAML key, but that's not necessary now.

---

## Phase 3: Model Upgrade (SigLIP)

**Impact**: Major quality improvement (~10% better zero-shot accuracy).
**Difficulty**: Medium
**Files**: `semantic.py`, `pyproject.toml`, possibly new `clipshow/detection/models.py`

### What Changes

Replace CLIP ViT-B/32 with SigLIP ViT-B/16 as the default model. SigLIP uses sigmoid loss instead of contrastive loss, producing better-calibrated similarity scores and stronger zero-shot performance (~70% ImageNet zero-shot vs CLIP ViT-B/32's ~63%).

### Approach

The key challenge is that `onnx_clip` only ships CLIP ViT-B/32. We have two options:

**Option A: Use `open_clip` ONNX exports** (recommended)
- Export SigLIP ViT-B/16 from HuggingFace/`open_clip` to ONNX format
- Host the exported ONNX files (image encoder + text encoder) in a GitHub release or CDN
- Write a thin model loader in `clipshow/detection/models.py` that downloads and caches models, then creates ONNX Runtime sessions directly (bypassing `onnx_clip` entirely)
- This gives us full control over model selection and removes the `onnx_clip` dependency

**Option B: Keep `onnx_clip`, add SigLIP alongside**
- Use `open_clip` at runtime to load SigLIP, but this pulls in PyTorch (~2GB) — unacceptable for a lightweight desktop app

Option A is strongly preferred. The new model loader would look like:

```python
# clipshow/detection/models.py

MODELS = {
    "clip-vit-b-32": {
        "image_encoder": "clip-vit-b-32-image.onnx",
        "text_encoder": "clip-vit-b-32-text.onnx",
        "image_size": 224,
        "embed_dim": 512,
        "url_base": "https://github.com/.../releases/download/models-v1/",
    },
    "siglip-vit-b-16": {
        "image_encoder": "siglip-vit-b-16-image.onnx",
        "text_encoder": "siglip-vit-b-16-text.onnx",
        "image_size": 224,
        "embed_dim": 768,
        "url_base": "https://github.com/.../releases/download/models-v1/",
    },
}

class VisionLanguageModel:
    """Thin ONNX Runtime wrapper for CLIP-family models."""

    def __init__(self, model_name: str = "siglip-vit-b-16"):
        ...

    def get_image_embeddings(self, images: list[Image]) -> np.ndarray:
        # Preprocess: resize, normalize, stack into batch tensor
        # Run ONNX image encoder session
        ...

    def get_text_embeddings(self, texts: list[str]) -> np.ndarray:
        # Tokenize with the model's tokenizer
        # Run ONNX text encoder session
        ...
```

This replaces the `onnx_clip` dependency with direct `onnxruntime` usage plus a small preprocessing/tokenization layer. The preprocessing (resize, center crop, normalize to ImageNet stats) is straightforward numpy/PIL. Tokenization can use HuggingFace's `tokenizers` (fast Rust-based, ~5MB wheel, no PyTorch).

### Config Changes

- Add `semantic_model: str = "siglip-vit-b-16"` to `Settings`
- Add `semantic.model` to YAML config
- Update the YAML example in the README
- Keep "clip-vit-b-32" as an option for users who want smaller downloads

### Risks

- ONNX export process: SigLIP export may need custom handling for sigmoid vs softmax scoring. Need to verify the exported model produces correct similarity scores.
- Tokenizer: SigLIP uses a SentencePiece tokenizer, not the CLIP BPE tokenizer. Need to bundle or download the tokenizer vocab alongside the model.
- Model hosting: Need a reliable place to host ~400MB ONNX files. GitHub Releases has a 2GB per-file limit, which is fine.

### Migration

- Default to SigLIP for new installations
- Existing users keep their current model until they explicitly change it or reset settings
- Sigmoid normalization parameters (`_SIGMOID_CENTER`, `_SIGMOID_SCALE`) will need retuning for SigLIP's different score distribution

---

## Phase 4: Adaptive Sampling

**Impact**: 2-3x speed improvement on long videos with sparse interesting content + better temporal precision on interesting sections.
**Difficulty**: Medium
**Files**: `semantic.py`

### What Changes

Replace fixed 2 FPS sampling with a two-pass approach:

1. **Coarse pass** at 1 FPS using a cheap heuristic (histogram difference between consecutive frames) to identify "active" regions.
2. **Fine pass** at 4-8 FPS using the full CLIP model, but only on active regions plus a small margin.

For a 10-minute video that's 80% static establishing shots, this processes ~120 frames instead of ~1200 — a 10x reduction in CLIP inference calls.

### Approach

```python
def detect(self, video_path, ...):
    # Pass 1: Coarse activity detection (cheap, no model needed)
    activity_mask = self._coarse_activity_scan(video_path)

    # Pass 2: CLIP inference only on active regions
    model = self._model or self._load_model()
    ...
    for frame_idx, frame in sampled_frames:
        t = frame_idx / fps
        if not activity_mask.is_active(t):
            continue  # Skip boring regions entirely
        # Full CLIP inference at higher FPS in active regions
        ...

def _coarse_activity_scan(self, video_path) -> ActivityMask:
    """Fast first pass: histogram difference at 1 FPS."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(1, int(fps))  # 1 FPS

    prev_hist = None
    activity = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
            hist = hist.flatten() / hist.sum()
            if prev_hist is not None:
                diff = np.sum(np.abs(hist - prev_hist))
                activity.append((frame_idx / fps, diff))
            prev_hist = hist
        frame_idx += 1
    cap.release()
    # Mark regions where activity > threshold (+ margin) as active
    return ActivityMask(activity, threshold=0.05, margin_sec=2.0)
```

The coarse scan adds ~2-5 seconds for a 5-minute video (just OpenCV histogram, no model inference). The savings come from skipping the expensive CLIP inference on 60-80% of frames in typical home videos.

### Config Changes

- Add `semantic_adaptive_sampling: bool = True` to `Settings`
- Add `semantic.adaptive_sampling` to YAML config
- Keep a fallback to fixed-rate sampling when disabled

---

## Phase 5: Multi-Scale Crops

**Impact**: Better detection of small but important subjects (faces, text, objects).
**Difficulty**: Medium
**Files**: `semantic.py`

### What Changes

Currently only the full frame is embedded. Small but semantically important regions (a person's face, a sign, a small animal) get diluted in the 224x224 resize. Adding a center crop alongside the full frame captures subjects that tend to be in the middle of the shot.

### Approach

For each sampled frame, generate two views:
1. **Full frame** — resized to 224x224 (current behavior)
2. **Center crop** — crop the center 50% of the frame, then resize to 224x224

Score both views and take the max:

```python
def _prepare_views(self, img: Image) -> list[Image]:
    """Generate multi-scale views of a frame."""
    views = [img]  # Full frame (model handles resize)

    # Center crop: 50% of the shorter dimension
    w, h = img.size
    crop_size = min(w, h) // 2
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    center = img.crop((left, top, left + crop_size, top + crop_size))
    views.append(center)

    return views
```

Then in the scoring loop, embed both views and take the max similarity:

```python
views = self._prepare_views(img)
image_embeddings = model.get_image_embeddings(views)  # (2, 512)
pos_sims = (image_embeddings @ pos_emb.T).max(axis=1)  # (2, P)
neg_sims = (image_embeddings @ neg_emb.T).max(axis=1)
# Take best score across views
raw = (pos_sims.max(axis=0) - neg_sims.max(axis=0)).max()
```

This doubles inference cost per frame, but combined with batching (Phase 1) and adaptive sampling (Phase 4), the net throughput impact is modest. The center crop can be batched alongside full frames — with batch_size=16, we process 8 frames × 2 views = 16 images per batch.

### Config Changes

- Add `semantic_multi_scale: bool = True` to `Settings`
- Add `semantic.multi_scale` to YAML config

---

## Phase 6: Temporal Context (Sliding Window)

**Impact**: Better detection of actions and events that span multiple frames.
**Difficulty**: Medium-High
**Files**: `semantic.py`, possibly `scoring.py`

### What Changes

Current scoring is purely per-frame — a frame either matches a prompt or it doesn't. Actions like "people laughing" or "dancing" unfold over time and may not be recognizable in any single frame. Pooling embeddings over a small temporal window captures short-range dynamics.

### Approach

For each scoring position, average the CLIP embeddings of 3-5 neighboring frames before computing similarity:

```python
TEMPORAL_WINDOW = 5  # frames (at 2-4 FPS = 1-2.5 seconds of context)

# After computing all frame embeddings for a batch/chunk:
# frame_embeddings: (T, 512) where T = number of sampled frames

def _temporal_pool(self, embeddings: np.ndarray, window: int) -> np.ndarray:
    """Mean-pool embeddings over a sliding window."""
    T, D = embeddings.shape
    pooled = np.zeros_like(embeddings)
    half = window // 2
    for i in range(T):
        start = max(0, i - half)
        end = min(T, i + half + 1)
        pooled[i] = embeddings[start:end].mean(axis=0)
        # Re-normalize to unit length for cosine similarity
        pooled[i] /= np.linalg.norm(pooled[i])
    return pooled
```

This requires a two-pass approach within the detector: first compute all frame embeddings, then pool, then score. This changes the current streaming architecture slightly — we need to store all embeddings before scoring instead of scoring as we go. Memory cost: for a 5-minute video at 2 FPS, that's 600 frames × 512 floats × 4 bytes = ~1.2 MB. Negligible.

### Dependencies

- Benefits most when combined with Phase 1 (batching) since we're already accumulating embeddings.
- Should be implemented after Phase 1.

### Config Changes

- Add `semantic_temporal_window: int = 5` to `Settings`
- Add `semantic.temporal_window` to YAML config
- 0 or 1 = disabled (per-frame scoring, current behavior)

---

## Implementation Order

| Phase | Description | Depends On | Estimated Effort |
|-------|-------------|------------|------------------|
| 1 | Batch processing | — | Small (1 session) |
| 2 | Prompt ensemble | — | Small (1 session) |
| 3 | Model upgrade (SigLIP) | — | Large (2-3 sessions) |
| 4 | Adaptive sampling | — | Medium (1-2 sessions) |
| 5 | Multi-scale crops | Phase 1 | Small (1 session) |
| 6 | Temporal context | Phase 1 | Medium (1-2 sessions) |

Phases 1 and 2 are independent and can be done in either order. They're the lowest-hanging fruit — easy to implement with clear payoff.

Phase 3 is the biggest quality win but requires the most infrastructure work (new model loader, tokenizer, ONNX export pipeline, model hosting).

Phases 4, 5, and 6 build on the batching foundation from Phase 1 and can be done in any order after it.

### Suggested beads breakdown

Each phase becomes its own bead, with dependency edges:

```
Phase 1 (batch) ──┬──> Phase 5 (multi-scale crops)
                  └──> Phase 6 (temporal context)
Phase 2 (prompt ensemble)
Phase 3 (model upgrade)
Phase 4 (adaptive sampling)
```

## Testing Strategy

Each phase gets its own test additions:

- **Phase 1**: Verify batched inference produces identical scores to frame-by-frame (within float tolerance). Benchmark speed on a synthetic video.
- **Phase 2**: Verify ensembled embeddings have unit norm. Verify scores change (are not identical to non-ensembled) on a test frame.
- **Phase 3**: Integration test with new model loader — download, cache, run inference, verify output shape and score range. Mock download for CI.
- **Phase 4**: Verify coarse scan correctly identifies active vs static regions on synthetic video. Verify total frames processed is lower than fixed-rate.
- **Phase 5**: Verify multi-scale produces 2 views per frame. Verify max-across-views scoring.
- **Phase 6**: Verify temporal pooling output shape matches input. Verify pooled embeddings differ from unpooled on sequences with variation.

All tests use the existing synthetic video fixtures from `conftest.py`. Semantic detector tests should mock the ONNX model to avoid requiring the 338MB download in CI.

## Open Questions (Phases 1-6)

1. **SigLIP ONNX export**: Has anyone published pre-exported SigLIP ONNX files, or do we need to do the export ourselves? If we export, we need a one-time script that requires PyTorch (not shipped with ClipShow, just used for the export).

2. **`onnx_clip` batch support**: Does `onnx_clip.OnnxClip(batch_size=N)` actually support N > 1 properly, or does it just set a fixed input dimension? Need to test before Phase 1.

3. **Sigmoid retuning**: SigLIP produces different raw score distributions than CLIP. The current `_SIGMOID_CENTER = 0.25` and `_SIGMOID_SCALE = 20.0` were tuned for CLIP ViT-B/32. Phase 3 needs a calibration step to find good values for SigLIP. This could be done empirically on a small set of test videos with known interesting moments.

4. **Drop `onnx_clip` dependency?**: If we build our own model loader in Phase 3, the `onnx_clip` package becomes redundant. We could remove it, simplifying dependencies to just `onnxruntime` + `tokenizers`. This would also let us support batch_size > 1 natively without worrying about `onnx_clip`'s limitations.

---
---

# Quality Step-Change: Beyond CLIP

Phases 1-6 are incremental improvements to a fundamentally limited architecture. CLIP ViT-B/32 was trained on static image-caption pairs in 2021 — there's a ceiling to how good it can get. For a ~5x quality jump, we need to move to models with fundamentally better understanding.

Four options were evaluated. **Option 4 (audio-visual multimodal) is the preferred first step**, followed by Option 3 (local VLM) as future work.

### Option 1: API-based VLM scoring (future reference)

Send keyframes to a cloud VLM (Claude, Gemini, GPT-4o) and ask it to rate interestingness. Best quality available but costs $1-15 per video, requires internet, and is slow without parallelization. Could be a premium/optional mode.

### Option 2: Hybrid CLIP + API VLM (future reference)

Use CLIP as a cheap local filter to find the top 10-20% candidate frames, then send only those to an API VLM. Reduces cost to ~$0.50-2 per video. Nearly as good as full API scoring since CLIP's false negatives are rare.

### Option 3: Local small VLM (future work)

Run a 2B-parameter vision-language model (Qwen2-VL-2B, InternVL2-2B) locally via ONNX or llama.cpp. 10-50x better scene understanding than CLIP. Requires GPU for reasonable speed (~0.5 FPS on CPU). Model is 2-4GB. This is the next quality frontier after Option 4.

### Option 4: Audio-visual multimodal (ACTIVE — see below)

Use a model trained on video+audio+text jointly. Understands that cheering + fast motion = exciting. Better "interestingness" detection than pure vision since it fuses modalities the way humans perceive video.

---

## Phase 7: Audio-Visual Multimodal Detector (LanguageBind)

**Impact**: Step-change in quality — the model jointly understands audio and visual content against text prompts, so "crowd cheering at a goal" is one concept, not two disconnected signals (audio peak + motion) combined via weighted average.
**Difficulty**: High
**Priority**: Active — preferred next major improvement

### Background: Model Selection

| Model | Params | Size | Audio Perf | Video Perf | ONNX | PyTorch Required |
|-------|--------|------|------------|------------|------|------------------|
| ImageBind (Meta) | 1.07B | 4.8GB | Baseline | Baseline | No (export needed) | At runtime — no |
| LanguageBind | ~400M per encoder | ~1.5-2GB total | +24% vs ImageBind on ESC-50 | +14% on K400 | No (export needed) | At runtime — no |
| CLAP (LAION) | ~150M | ~600MB | Good (audio-text only) | N/A | Partial | No |

**LanguageBind is the best fit** because:
- Language-centric binding: aligns audio, video, and text to a shared embedding space through language (not through images like ImageBind), which produces better text-queryable embeddings
- Significantly better audio understanding than ImageBind (+24% on ESC-50, +10% on AudioSet)
- Modular encoder architecture: separate audio, video, and text encoders that can be exported to ONNX individually
- Each encoder is smaller than the full ImageBind model

**Key constraint**: LanguageBind (like ImageBind) requires PyTorch for model loading. Our approach is to **export the encoders to ONNX once** (dev-time script requiring PyTorch) and **ship only ONNX files** at runtime (no PyTorch dependency for users). This follows the same pattern proposed in Phase 3 for SigLIP.

### Architecture

A new `AudioVisualDetector` class that sits alongside (not replaces) the existing `SemanticDetector`:

```
clipshow/detection/
├── semantic.py          # Existing CLIP-based visual-only detector
├── audiovisual.py       # NEW: LanguageBind audio+video+text detector
├── models.py            # NEW: Shared ONNX model download/cache/load
└── pipeline.py          # Updated: register audiovisual detector
```

The detector processes both the video frames AND the audio track jointly:

```
Video file
    ├─ Video frames (sampled at 2 FPS)
    │   ↓
    │   LanguageBind Video Encoder (ONNX) → video_emb (N, D)
    │
    ├─ Audio track (extracted via ffmpeg)
    │   ↓
    │   Split into 2-second windows, mel-spectrogram
    │   ↓
    │   LanguageBind Audio Encoder (ONNX) → audio_emb (M, D)
    │
    └─ Text prompts
        ↓
        LanguageBind Text Encoder (ONNX) → text_emb (P, D)

Scoring:
    video_sim = video_emb @ text_emb.T    # (N, P)
    audio_sim = audio_emb @ text_emb.T    # (M, P)

    # Fuse: for each time position, combine video and audio similarities
    combined_sim = α * video_sim + (1-α) * audio_sim   # α=0.6 default

    # Take best prompt match per time step
    score = combined_sim.max(axis=1)      # (T,)
```

This is fundamentally different from the current pipeline approach where audio and video are scored separately by independent detectors and combined via `weighted_combine()` in `scoring.py`. Here the fusion happens in the *embedding space* — the model was trained to put "crowd cheering" audio and "crowd celebrating" video near the same text, so fusing before the text comparison captures cross-modal semantics that post-hoc weighted averaging cannot.

### Implementation Steps

#### Step 7a: ONNX Model Infrastructure (`models.py`)

Shared model download/cache/load system used by both the SigLIP upgrade (Phase 3) and LanguageBind. This is foundational infrastructure.

```python
# clipshow/detection/models.py

MODEL_REGISTRY = {
    # Phase 3
    "siglip-vit-b-16": { ... },
    # Phase 7
    "languagebind-video": {
        "file": "languagebind-video-encoder.onnx",
        "size_mb": 600,
        "url": "https://github.com/.../releases/download/models-v2/...",
    },
    "languagebind-audio": {
        "file": "languagebind-audio-encoder.onnx",
        "size_mb": 400,
        "url": "https://github.com/.../releases/download/models-v2/...",
    },
    "languagebind-text": {
        "file": "languagebind-text-encoder.onnx",
        "size_mb": 500,
        "url": "https://github.com/.../releases/download/models-v2/...",
    },
}

class ModelManager:
    """Download, cache, and load ONNX models."""

    cache_dir = Path.home() / ".clipshow" / "models"

    def ensure_model(self, name: str, progress_cb=None) -> Path:
        """Download model if not cached, return local path."""
        ...

    def load_session(self, name: str, **kwargs) -> ort.InferenceSession:
        """Load an ONNX Runtime session for a model."""
        path = self.ensure_model(name)
        providers = self._get_providers()  # CUDA > CoreML > CPU
        return ort.InferenceSession(str(path), providers=providers)

    def _get_providers(self) -> list[str]:
        """Detect available execution providers, prefer GPU."""
        available = ort.get_available_providers()
        preferred = ["CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"]
        return [p for p in preferred if p in available]
```

#### Step 7b: ONNX Export Script (`scripts/export_languagebind.py`)

One-time developer script (not shipped with ClipShow) that exports LanguageBind encoders to ONNX. Requires PyTorch + LanguageBind installed in a separate dev environment.

```python
# scripts/export_languagebind.py
"""Export LanguageBind encoders to ONNX format.

Requires: pip install torch languagebind onnx onnxruntime

Usage: python scripts/export_languagebind.py --output-dir ./exported_models/
"""

def export_video_encoder(output_path):
    """Export LanguageBind video encoder."""
    # Load pretrained LanguageBind_Video_FT
    # Trace with dummy input (B, 3, 8, 224, 224) for 8-frame video
    # Export via torch.onnx.export with dynamic batch axis
    ...

def export_audio_encoder(output_path):
    """Export LanguageBind audio encoder."""
    # Load pretrained LanguageBind_Audio_FT
    # Trace with dummy mel-spectrogram input
    # Export via torch.onnx.export
    ...

def export_text_encoder(output_path):
    """Export LanguageBind text encoder (shared across modalities)."""
    # Load pretrained text encoder
    # Trace with dummy token IDs
    # Export via torch.onnx.export
    ...

def validate_exports(output_dir):
    """Run inference on exported models, compare to PyTorch outputs."""
    # Load both PyTorch and ONNX versions
    # Run same inputs through both
    # Assert outputs match within tolerance
    ...
```

The exported ONNX files get uploaded to a GitHub release as binary assets. Users download them on first use via `ModelManager.ensure_model()`.

#### Step 7c: Audio Preprocessing

LanguageBind's audio encoder expects mel-spectrograms. We need a preprocessing pipeline that:

1. Extracts audio from the video file (reuse existing ffmpeg path from `AudioDetector`)
2. Splits into 2-second windows aligned with the video sample times
3. Computes mel-spectrograms using `librosa` (already a dependency)

```python
def _extract_audio_features(self, video_path: str, duration: float) -> np.ndarray:
    """Extract mel-spectrogram windows aligned to video sample times."""
    # 1. Extract audio to temp WAV via ffmpeg (same as AudioDetector)
    audio, sr = librosa.load(tmp_wav, sr=16000)

    # 2. Split into 2-second windows at SAMPLE_FPS intervals
    window_samples = 2 * sr  # 32000 samples per window
    step_samples = int(sr / SAMPLE_FPS)

    windows = []
    for t in sample_times:
        center = int(t * sr)
        start = max(0, center - window_samples // 2)
        end = start + window_samples
        chunk = audio[start:end]
        if len(chunk) < window_samples:
            chunk = np.pad(chunk, (0, window_samples - len(chunk)))

        # 3. Mel-spectrogram (128 mel bands, matching LanguageBind training)
        mel = librosa.feature.melspectrogram(
            y=chunk, sr=sr, n_mels=128, n_fft=1024, hop_length=320
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        windows.append(mel_db)

    return np.stack(windows)  # (T, 128, freq_bins)
```

This reuses `librosa` (already required by `AudioDetector`) so adds no new dependency.

#### Step 7d: AudioVisualDetector Implementation

```python
# clipshow/detection/audiovisual.py

class AudioVisualDetector(Detector):
    """LanguageBind-based audio-visual-text detector.

    Jointly scores video frames and audio against text prompts using
    LanguageBind's shared embedding space. Unlike SemanticDetector (which
    only sees video) and AudioDetector (which only sees audio), this detector
    understands cross-modal semantics: a cheering crowd sounds exciting AND
    looks exciting, and the model was trained to fuse those signals.
    """

    name = "audiovisual"

    def __init__(
        self,
        prompts: list[str] | None = None,
        negative_prompts: list[str] | None = None,
        audio_weight: float = 0.4,      # Weight of audio vs video in fusion
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
        self._video_session = manager.load_session("languagebind-video", ...)
        self._audio_session = manager.load_session("languagebind-audio", ...)
        self._text_session = manager.load_session("languagebind-text", ...)

    def detect(self, video_path, progress_callback=None, cancel_flag=None):
        if not self._video_session:
            self._load_models(progress_callback)

        # 1. Sample video frames (same as SemanticDetector)
        frames, frame_times = self._sample_frames(video_path)

        # 2. Extract audio mel-spectrograms aligned to frame times
        audio_features, audio_times = self._extract_audio_features(video_path)

        # 3. Encode all modalities
        video_emb = self._encode_video(frames)          # (N, D)
        audio_emb = self._encode_audio(audio_features)  # (M, D)
        text_pos = self._encode_text(self._prompts)     # (P, D)
        text_neg = self._encode_text(self._negative_prompts)

        # 4. Score: fuse video + audio similarities before text comparison
        video_pos = video_emb @ text_pos.T              # (N, P)
        video_neg = video_emb @ text_neg.T
        audio_pos = self._align_to_video(audio_emb @ text_pos.T, audio_times, frame_times)
        audio_neg = self._align_to_video(audio_emb @ text_neg.T, audio_times, frame_times)

        α = 1.0 - self._audio_weight
        fused_pos = α * video_pos + (1 - α) * audio_pos
        fused_neg = α * video_neg + (1 - α) * audio_neg

        raw = fused_pos.max(axis=1) - fused_neg.max(axis=1)
        scores = sigmoid_normalize(raw)

        # 5. Smooth and return
        ...
```

#### Step 7e: Pipeline Integration

Register the new detector in `pipeline.py`:

```python
# In _get_optional_detector():
elif name == "audiovisual":
    try:
        from clipshow.detection.audiovisual import AudioVisualDetector
        return AudioVisualDetector
    except ImportError:
        return None

# In Settings:
audiovisual_weight: float = 0.0  # disabled by default (requires model download)
```

Add to the UI analyze panel as a new detector row with a note about the ~1.5GB model download. Add `audiovisual` to the YAML `detectors:` block.

### Config Changes

```python
# config.py additions
audiovisual_weight: float = 0.0
audiovisual_audio_weight: float = 0.4  # audio vs video balance in fusion
```

```yaml
# YAML additions
detectors:
  audiovisual: 0.3

audiovisual:
  audio_weight: 0.4
  prompts: [...]
  negative_prompts: [...]
```

### Risks & Mitigations

1. **ONNX export complexity**: LanguageBind uses custom video/audio preprocessing that may not trace cleanly to ONNX. *Mitigation*: Export only the encoder transformers, keep preprocessing in Python (numpy/librosa/PIL).

2. **Model size**: ~1.5GB total for three encoders. *Mitigation*: Download on first use with progress indicator. Cache in `~/.clipshow/models/`. Users who don't enable the detector never download anything.

3. **CPU speed**: 400M+ params per encoder is slow on CPU. *Mitigation*: Auto-detect GPU via ONNX Runtime execution providers (CUDA, CoreML). Warn users that CPU-only will be slow. Batch processing (Phase 1 approach) helps. Adaptive sampling (Phase 4) reduces frame count.

4. **LanguageBind video encoder expects 8 frames**: The video encoder was trained on 8-frame clips (not single frames). *Mitigation*: For each sample time, grab 8 frames spanning ~2 seconds centered on that time. This naturally provides temporal context (free Phase 6).

5. **Tokenizer dependency**: LanguageBind uses a specific tokenizer. *Mitigation*: Bundle the tokenizer vocab file alongside the ONNX models. Use HuggingFace `tokenizers` (Rust-based, fast, small) at runtime.

### Fallback: CLAP + SigLIP Fusion

If the LanguageBind ONNX export proves too complex (risk #1), a simpler fallback achieves ~70% of the benefit:

- Use **LAION-CLAP** (~600MB, has partial ONNX support via `msclap`) for audio-text scoring
- Use **SigLIP** (Phase 3) for video-text scoring
- Fuse scores in `AudioVisualDetector` the same way — just with two independently trained models instead of one jointly trained model

This is less powerful (the models don't share an embedding space, so cross-modal semantics are lost) but still better than the current approach of combining separate `AudioDetector` and `SemanticDetector` scores via `weighted_combine()`, because CLAP actually understands audio *content* (speech, music genre, environmental sounds) rather than just amplitude/onset peaks.

### Testing Strategy

- **Export validation**: Script that compares ONNX output to PyTorch output on reference inputs (part of export script, not CI)
- **Unit tests**: Mock ONNX sessions, verify correct preprocessing shapes, verify fusion math, verify score normalization
- **Integration test**: Load real ONNX models (small test fixtures, not full models), verify end-to-end on synthetic video with audio
- **CI**: Mock all ONNX inference (models too large for CI). Test preprocessing, fusion logic, and pipeline integration with canned embeddings

---

## Phase 8 (Future): Local Small VLM

After Phase 7, the next quality frontier is running a small vision-language model (Qwen2-VL-2B, InternVL2-2B) locally. These models can answer "is this an interesting moment?" with genuine scene understanding rather than embedding similarity. This is a larger effort tracked separately.

---

## Revised Implementation Order

| Phase | Description | Depends On | Priority |
|-------|-------------|------------|----------|
| 1 | Batch processing | — | High (quick win) |
| 2 | Prompt ensemble | — | High (quick win) |
| 7a | ONNX model infrastructure | — | **High (active)** |
| 7b | LanguageBind ONNX export | 7a | **High (active)** |
| 7c | Audio preprocessing | 7b | **High (active)** |
| 7d | AudioVisualDetector | 7a, 7b, 7c | **High (active)** |
| 7e | Pipeline + UI integration | 7d | **High (active)** |
| 3 | Model upgrade (SigLIP) | 7a | Medium |
| 4 | Adaptive sampling | — | Medium |
| 5 | Multi-scale crops | 1 | Medium |
| 6 | Temporal context | 1 | Medium |
| 8 | Local small VLM | 7a | Future |

Phases 1 and 2 are independent quick wins that can happen in parallel with Phase 7 work.

Phase 7a (model infrastructure) is foundational — Phase 3 (SigLIP) reuses it.

Phases 7b-7e are the core LanguageBind integration, done sequentially.
