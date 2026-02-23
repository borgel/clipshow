# ClipShow - Automatic Highlight Reel Generator

## Context

Build a desktop app that takes one or more video clips (from phones, Meta Ray-Ban glasses, etc.), automatically detects "interesting moments" using scene changes, audio peaks, motion, and semantic content analysis, then assembles a highlight reel. Users can preview/reorder/trim clips or run fully automatic. Must package for macOS and Windows with fully automated CI/CD releases.

## Tech Stack

- **UI**: PySide6 (LGPL Qt6 bindings)
- **Scene detection**: PySceneDetect
- **Audio analysis**: librosa (onset strength, RMS energy)
- **Motion detection**: OpenCV frame differencing
- **Semantic detection** (optional): `onnx_clip` for CLIP-based content scoring, MediaPipe + `deepface-onnx` for face/emotion detection — all ONNX Runtime based, no PyTorch dependency
- **Video assembly**: MoviePy 2.x (FFmpeg backend)
- **Video preview**: PySide6 QMediaPlayer + QVideoWidget
- **Packaging**: PyInstaller, automated via GitHub Actions

## Project Structure

```
clipshow/
├── pyproject.toml
├── .github/
│   └── workflows/
│       ├── ci.yml                   # Lint + test on every push/PR
│       └── release.yml              # Build + publish on version tags
├── clipshow/
│   ├── __init__.py
│   ├── __main__.py                  # Entry point, CLI arg parsing
│   ├── app.py                       # QApplication setup + auto-mode runner
│   ├── config.py                    # Settings dataclass with JSON persistence
│   ├── model/
│   │   ├── project.py               # Project, VideoSource, ExportSettings
│   │   ├── moments.py               # DetectedMoment, HighlightSegment
│   │   └── signals.py               # Qt signals for cross-component updates
│   ├── detection/
│   │   ├── base.py                  # Abstract Detector + DetectorResult
│   │   ├── scene.py                 # PySceneDetect wrapper
│   │   ├── audio.py                 # librosa onset/RMS analysis
│   │   ├── motion.py                # OpenCV frame differencing
│   │   ├── semantic.py              # CLIP content scoring (optional, lazy-loaded)
│   │   ├── emotion.py               # MediaPipe face + emotion detection (optional, lazy-loaded)
│   │   ├── pipeline.py              # Runs all enabled detectors, delegates to scoring
│   │   └── scoring.py               # Weighted combination, thresholding, segment merging
│   ├── export/
│   │   ├── assembler.py             # MoviePy clip concatenation + encoding
│   │   └── ffprobe.py               # Video metadata extraction
│   ├── ui/
│   │   ├── main_window.py           # 4-step tabbed workflow
│   │   ├── import_panel.py          # Drag-drop file import + file list
│   │   ├── analyze_panel.py         # Detector settings + progress bars
│   │   ├── review_panel.py          # Segment list + preview + trim controls
│   │   ├── export_panel.py          # Output settings + progress
│   │   ├── video_preview.py         # QMediaPlayer/QVideoWidget wrapper
│   │   ├── timeline_widget.py       # Visual bar showing detected moments
│   │   └── segment_list.py          # Draggable reorderable segment list
│   └── workers/
│       ├── analysis_worker.py       # QThread for detection pipeline
│       └── export_worker.py         # QThread for video encoding
├── resources/icons/                 # .icns, .ico, .png app icons
├── packaging/
│   ├── clipshow_macos.spec          # PyInstaller macOS .app bundle
│   ├── clipshow_windows.spec        # PyInstaller Windows .exe
│   └── installer.iss               # Inno Setup script for Windows installer
└── tests/
    ├── conftest.py                  # Fixtures: synthetic videos, offscreen Qt setup
    ├── baselines/                   # Visual regression baseline screenshots
    ├── test_scoring.py              # Unit: score math (pure numpy, fast)
    ├── test_pipeline.py             # Unit: pipeline with mock detectors
    ├── test_ffprobe.py              # Integration: metadata extraction
    ├── test_detectors.py            # Integration: each detector on synthetic video
    ├── test_assembler.py            # Integration: clip assembly
    ├── test_ui_import.py            # UI: import panel interactions
    ├── test_ui_analyze.py           # UI: analyze panel + worker signals
    ├── test_ui_review.py            # UI: segment list, trim, reorder
    ├── test_ui_export.py            # UI: export settings + worker
    ├── test_ui_layout.py            # UI: visual regression screenshots
    ├── test_auto_mode.py            # E2E: full auto pipeline
    └── test_ui_workflow.py          # E2E: full UI journey with qtbot
```

## Core Architecture

### Detection Pipeline

Each detector independently scores a video over time (one float per ~0.1s timestep, normalized 0.0-1.0). All detectors implement the same `Detector` base class and are pluggable:

**Core detectors** (always available, lightweight):

1. **SceneDetector** — PySceneDetect `ContentDetector` HSV-weighted frame differences. Spikes at cuts/transitions.
2. **AudioDetector** — Extract audio to temp WAV via FFmpeg, run librosa onset strength + RMS energy. Captures loud moments, speech, beats.
3. **MotionDetector** — OpenCV `absdiff` on decimated grayscale frames. Captures action/movement.

**Semantic detectors** (optional, lazy-loaded on first use, download models automatically):

4. **SemanticDetector** (`detection/semantic.py`) — Uses `onnx_clip` (ONNX Runtime, no PyTorch) with CLIP ViT-B/32 (338MB model, downloaded to `~/.clipshow/models/` on first use). Samples frames at 1 FPS to stay fast. Scores each frame against text prompts like "exciting moment", "people laughing", "beautiful scenery", "action scene". Returns cosine similarity as score. Users can customize the text prompts. ~60-150s for a 5-min video on CPU.
5. **EmotionDetector** (`detection/emotion.py`) — Uses MediaPipe for fast face detection (30+ FPS on CPU, tiny model) then `deepface-onnx` for emotion classification (no TensorFlow). Scores frames higher when faces show positive/high-energy emotions (happy, surprise). Samples at 3 FPS. ~30-50s for a 5-min video on CPU.

**Lazy loading pattern**: Core app imports nothing from onnx_clip/deepface-onnx/mediapipe at startup. When a user enables a semantic detector, we `importlib.import_module()` it on demand. If the package isn't installed, the UI shows a "Install optional dependencies?" prompt. Models download automatically on first use with a progress indicator.

### Segment Selection (`scoring.py`)

This is the algorithmic heart:

1. Resample all detector results to common 10 samples/sec time base
2. Weighted combination of all enabled detectors, renormalize to [0,1]
3. Threshold: mark timesteps where `combined >= threshold` as interesting
4. Extract contiguous runs → candidate segments
5. Apply user-configured pre/post padding, clamp to video boundaries
6. Merge overlapping or nearly-adjacent segments (gap < 0.5s)
7. Filter: discard segments shorter than `min_duration`, cap at `max_duration`
8. Rank by peak score

### Data Flow

```
Import files → ffprobe metadata → Project.sources
                                       ↓
Analyze → DetectionPipeline per source → DetectedMoments
                                              ↓
Convert → HighlightSegments (user-editable copies)
                                              ↓
Review → user reorders, trims, includes/excludes
                                              ↓
Export → MoviePy subclip + concatenate → output.mp4 (H.264)
```

### Key Data Classes

```python
@dataclass
class VideoSource:
    path: str
    duration: float  # from ffprobe
    width: int; height: int; fps: float; codec: str

@dataclass
class DetectedMoment:
    source_path: str
    start_time: float; end_time: float
    peak_score: float; mean_score: float
    contributing_detectors: list[str]

@dataclass
class HighlightSegment:
    source_path: str
    start_time: float; end_time: float  # user-adjustable
    score: float
    included: bool = True
    order: int = 0
```

### Settings (`config.py`)

User-adjustable, persisted to `~/.clipshow/settings.json`:

| Setting | Default | Purpose |
|---------|---------|---------|
| `scene_weight` | 0.3 | Weight for scene change detector |
| `audio_weight` | 0.25 | Weight for audio peak detector |
| `motion_weight` | 0.25 | Weight for motion detector |
| `semantic_weight` | 0.0 | Weight for CLIP semantic detector (0 = disabled) |
| `emotion_weight` | 0.2 | Weight for face/emotion detector (0 = disabled) |
| `semantic_prompts` | ["exciting moment", "people laughing", "beautiful scenery"] | Custom text prompts for CLIP scoring |
| `pre_padding_sec` | 1.0 | Seconds to include before detected event |
| `post_padding_sec` | 1.5 | Seconds to include after detected event |
| `score_threshold` | 0.5 | Minimum combined score to consider interesting |
| `min_segment_duration_sec` | 1.0 | Discard shorter segments |
| `max_segment_duration_sec` | 15.0 | Cap segment length |
| `output_codec` | libx264 | Maximum compatibility output |
| `output_fps` | 30.0 | Output framerate |
| `output_bitrate` | 8M | Output quality |

## UI Design

### 4-Step Workflow (tabs, enabled sequentially)

**Step 1 — Import**: Drag-drop area + file browser. List shows filename, duration, resolution. Remove/clear buttons.

**Step 2 — Analyze**: Sliders for detector weights, threshold, padding. Enable/disable checkboxes per detector. Semantic detectors show a note if models need downloading. Per-file progress bars. "Analyze All" button.

**Step 3 — Review**: Left side: draggable segment list with checkboxes. Right side: QMediaPlayer video preview. Bottom: trim in/out controls with ±0.5s nudge buttons. Click a segment to preview it.

**Step 4 — Export**: Output path picker, resolution/fps/bitrate settings, summary (N segments, total duration), export button + progress bar.

### Automatic Mode

`clipshow --auto --output reel.mp4 video1.mp4 video2.mov`

By default, auto mode shows a **minimal progress window** (just a progress bar + cancel button, no full UI). Add `--headless` to suppress all GUI for scripting/batch use. Runs pipeline with default settings, takes all segments above threshold sorted chronologically, assembles and writes output. Exits with status code.

### Clip Ordering

Default order is **chronological** — segments sorted by timestamp within each source, sources interleaved by file order. User can drag-reorder in the Review panel.

### Transitions

**Hard cuts only** (no crossfades). Keeps it simple and fast. Can add transition options in a future version.

## Threading

| Operation | Thread | Why |
|-----------|--------|-----|
| UI rendering, interaction | Main thread | Qt requirement |
| ffprobe metadata | Main thread | Fast (<100ms/file) |
| Detection pipeline | `AnalysisWorker(QThread)` | Seconds-to-minutes per video |
| Video preview playback | Main thread | QMediaPlayer is Qt-managed |
| Export/encoding | `ExportWorker(QThread)` | Minutes, can't block UI |
| Model download | `AnalysisWorker` (before detection) | Shows progress in UI |

Progress via Qt signals. Cancellation via flag checked between files/segments.

## Packaging & CI/CD

### Dependencies

Core (always installed):
- `PySide6>=6.6`, `moviepy>=2.0`, `scenedetect[opencv]>=0.6.2`, `librosa>=0.10`, `opencv-python-headless>=4.8`, `numpy>=1.24`

Optional (for semantic detectors):
- `onnx_clip`, `onnxruntime` — CLIP semantic scoring (~50MB + 338MB model download)
- `mediapipe`, `deepface-onnx` — face/emotion detection (~25MB)

Use `opencv-python-headless` (not `opencv-python`) to avoid Qt platform plugin conflicts.

### FFmpeg

Require FFmpeg in PATH for v0.1 (document Homebrew/winget install). App checks on startup and shows a helpful error if missing.

### GitHub Actions — CI (`.github/workflows/ci.yml`)

Triggers on every push and PR. Matrix: `ubuntu-latest`, `macos-latest`, `windows-latest`:
- Install Python 3.11, pip install deps + `pytest-qt`
- Linux: use `GabrielBB/xvfb-action@v1` for virtual display
- macOS/Windows: set `QT_QPA_PLATFORM=offscreen`
- `ruff check` for linting
- `pytest tests/ -v --cov` for all tests (unit + UI + integration)

### GitHub Actions — Release (`.github/workflows/release.yml`)

Triggers on version tags (`v*`). Free for public repos (unlimited minutes on standard runners).

**Job 1: `create-release`** — Creates a GitHub Release from the tag.

**Job 2: `build-macos`** (runs on `macos-latest`, needs `create-release`):
- Checkout, setup Python 3.11, install deps + PyInstaller
- `brew install ffmpeg`
- `pyinstaller packaging/clipshow_macos.spec`
- Create `.dmg` using `create-dmg` (Homebrew tool)
- Upload `.dmg` to the GitHub Release

**Job 3: `build-windows`** (runs on `windows-latest`, needs `create-release`):
- Checkout, setup Python 3.11, install deps + PyInstaller
- `choco install ffmpeg -y`
- `pyinstaller packaging/clipshow_windows.spec`
- Create installer with Inno Setup (`packaging/installer.iss`)
- Upload `.exe` installer to the GitHub Release

macOS and Windows jobs run in parallel. Users download from the GitHub Releases page.

**Code signing** (future, when ready): macOS notarization requires Apple Developer Program ($99/yr). Store certificate + credentials as GitHub Secrets. Add `codesign` and `notarytool` steps to the macOS job. Not required for initial development — unsigned apps work with right-click → Open.

### PyInstaller Specs

- macOS: `.app` bundle, `console=False`, `NSHighResolutionCapable=True`, register `.mp4`/`.mov` document types
- Windows: folder-mode `.exe`, `console=False`, `.ico` icon
- Both: `hiddenimports` for `PySide6.QtMultimedia`, `PySide6.QtMultimediaWidgets`, `scenedetect` submodules
- Exclude: `tkinter`, `matplotlib` (not needed, saves space)

## Build Order

Incremental — each step includes its tests. After writing tests for a module, run **`testreview-gemini`** to have Gemini confirm coverage is adequate before moving on.

1. **Scaffold** — `pyproject.toml`, package init, `__main__.py`, `config.py`, all model dataclasses, `conftest.py` with synthetic video fixtures, `.github/workflows/ci.yml`. Get `pip install -e .` and CI green.
2. **ffprobe** — `export/ffprobe.py` + `test_ffprobe.py`. Verify metadata extraction on synthetic videos.
3. **Scoring math** — `detection/scoring.py` + `test_scoring.py`. Pure numpy, no video deps. Run `testreview-gemini`.
4. **Scene detector** — `detection/scene.py` + tests in `test_detectors.py`. Test on synthetic "scene_change" video.
5. **Audio detector** — `detection/audio.py` + tests in `test_detectors.py`. Test on synthetic "loud_moment" video.
6. **Motion detector** — `detection/motion.py` + tests in `test_detectors.py`. Test on synthetic "motion" video.
7. **Pipeline** — `detection/pipeline.py` + `test_pipeline.py`. Mock detectors + real synthetic video E2E. Run `testreview-gemini`.
8. **Assembler** — `export/assembler.py` + `test_assembler.py`. Verify concatenation produces valid MP4.
9. **Auto mode** — Wire `run_auto_mode()` in `app.py` + `test_auto_mode.py`. Full CLI pipeline with synthetic videos.
10. **UI shell** — `main_window.py` with empty tabs + `test_ui_layout.py` baselines. PySide6 launches headlessly.
11. **Import panel** — Drag-drop, file list + `test_ui_import.py`. Verify button states, file list updates.
12. **Video preview** — QMediaPlayer wrapper (mock playback in tests).
13. **Analyze panel** — Settings UI, worker threading + `test_ui_analyze.py`. Verify slider/checkbox binding, worker signal handling.
14. **Review panel** — Segment list, preview, trim + `test_ui_review.py`. Verify reorder, trim, include/exclude.
15. **Export panel** — Settings, worker + `test_ui_export.py`. Verify export completion flow.
16. **Full UI E2E** — `test_ui_workflow.py`. Simulate entire Import→Analyze→Review→Export journey. Run `testreview-gemini` on all UI tests.
17. **Semantic detector** — `detection/semantic.py` with lazy loading + model download.
18. **Emotion detector** — `detection/emotion.py` with lazy loading.
19. **Visual regression baselines** — Regenerate `test_ui_layout.py` baselines for all panels.
20. **Packaging** — PyInstaller specs, `release.yml`, Inno Setup script, test on both platforms via CI.

## Testing Strategy

All tests run automatically in CI with no human intervention. Tests use `pytest` + `pytest-qt` and run headlessly on all platforms.

### Test Files

```
tests/
├── conftest.py                # Shared fixtures: synthetic videos, QApplication, offscreen setup
├── test_scoring.py            # Unit: score combination, thresholding, merging, padding
├── test_pipeline.py           # Unit: pipeline orchestration with mock detectors
├── test_assembler.py          # Integration: assemble synthetic clips, verify output
├── test_ffprobe.py            # Integration: extract metadata from synthetic videos
├── test_detectors.py          # Integration: run each detector on short synthetic videos
├── test_ui_import.py          # UI: import panel drag-drop, file list, button states
├── test_ui_analyze.py         # UI: slider values, checkbox states, worker signal handling
├── test_ui_review.py          # UI: segment list reorder, trim controls, preview integration
├── test_ui_export.py          # UI: export settings, progress bar, worker completion
├── test_ui_layout.py          # UI: visual regression — screenshot comparison against baselines
├── test_auto_mode.py          # E2E: run full auto-mode pipeline with synthetic video
└── test_ui_workflow.py        # E2E: simulate full Import → Analyze → Review → Export flow
```

### Layer 1: Unit Tests (fast, no video files)

**`test_scoring.py`** — Pure numpy, tests the algorithmic core:
- Weighted combination of detector score arrays
- Thresholding extracts correct contiguous regions
- Padding extends segments and clamps to boundaries
- Overlapping segments merge correctly
- Duration filtering discards too-short/too-long segments
- Ranking sorts by peak score

**`test_pipeline.py`** — Mock detectors return canned `DetectorResult` objects:
- Pipeline runs only enabled detectors (respects weight=0)
- Pipeline passes progress callbacks correctly
- Cancellation flag stops processing between detectors

### Layer 2: Integration Tests (use short synthetic videos)

**`conftest.py`** fixtures:
```python
# Generates 0.5-2s synthetic test videos using MoviePy
# - "static" video: uniform color (no motion, no scene changes)
# - "scene_change" video: abrupt color change at 50% mark
# - "motion" video: moving bright object on dark background
# - "loud_moment" video: silence then burst of white noise audio
# Videos cached per session to avoid regeneration
```

**`test_detectors.py`**:
- SceneDetector finds the scene change in "scene_change" video, returns spike at correct timestamp
- MotionDetector scores "motion" video higher than "static" video
- AudioDetector finds the loud moment in "loud_moment" video
- Each detector returns `DetectorResult` with correct `time_step` and score range [0,1]

**`test_assembler.py`**:
- Assembles two 1s synthetic clips → output exists, duration ~2s (verified via ffprobe)
- Respects subclip start/end times (0.2-0.8s of a 1s clip → ~0.6s output)
- Output is valid MP4 (ffprobe returns codec=h264)

**`test_ffprobe.py`**:
- Extracts duration, resolution, FPS, codec from synthetic video
- Returns correct values for known test inputs

### Layer 3: Automated UI Tests (pytest-qt, no human)

All UI tests use `pytest-qt`'s `qtbot` fixture. Widgets are created programmatically. `QT_QPA_PLATFORM=offscreen` for headless rendering.

**`test_ui_import.py`**:
- Add files programmatically → file list shows correct count and filenames
- Remove file → list updates, file removed
- Clear all → list empty
- Buttons enable/disable correctly (Next disabled when no files)

**`test_ui_analyze.py`**:
- Slider changes update `Settings` values
- Checkbox disable sets weight to 0
- "Analyze" button launches `AnalysisWorker` (use `qtbot.waitSignal(worker.all_complete)`)
- Progress bar updates via signals
- Cancel stops the worker

**`test_ui_review.py`**:
- Segment list populated from `Project.highlight_segments`
- Drag reorder changes `segment.order` (simulate with `QListWidget` internal move)
- Checkbox toggle changes `segment.included`
- Trim controls adjust `segment.start_time` / `segment.end_time`
- Click segment triggers `VideoPreview.play_segment()` (mock QMediaPlayer to avoid actual playback)

**`test_ui_export.py`**:
- Export settings bind to `ExportSettings` model
- Export button launches `ExportWorker`
- Progress bar reaches 100% on completion (use `qtbot.waitSignal(worker.complete)`)
- Output path matches what user specified

### Layer 4: Visual Regression Tests (automated screenshot comparison)

**`test_ui_layout.py`**:
- Capture screenshots of each panel via `widget.grab()` at fixed 800x600 size
- Compare against baseline PNGs stored in `tests/baselines/`
- Use `scikit-image` SSIM (Structural Similarity Index) with threshold 0.95
- On first run: generate baselines automatically
- On CI: compare current vs baseline, fail if SSIM < 0.95
- Save diff screenshots as test artifacts on failure

Panels tested:
- Import panel (empty state, with files loaded)
- Analyze panel (default settings, custom settings)
- Review panel (with segments loaded)
- Export panel (default state, mid-export progress)
- Main window (tab layout, navigation buttons)

Baselines regenerated via `pytest tests/test_ui_layout.py --update-baselines` flag.

### Layer 5: End-to-End Tests (full workflow, automated)

**`test_auto_mode.py`**:
- Invoke `run_auto_mode()` with synthetic "scene_change" + "motion" test videos
- Verify: output file exists, is valid MP4, duration > 0, duration < sum of inputs
- Verify: at least one segment was detected and included

**`test_ui_workflow.py`** — Simulates complete user journey with `qtbot`:
1. Create `MainWindow`
2. Programmatically add test video files to import panel
3. Click "Next" → verify Analyze tab active
4. Click "Analyze All" → `qtbot.waitSignal(analysis_complete, timeout=30000)`
5. Click "Next" → verify Review tab active, segment list non-empty
6. Toggle a segment's checkbox, reorder two segments
7. Click "Next" → verify Export tab active
8. Set output path to temp file, click "Export" → `qtbot.waitSignal(export_complete, timeout=30000)`
9. Verify output file exists and is valid MP4

### CI Headless Setup

```yaml
# In ci.yml:
- name: Run tests (Linux)
  uses: GabrielBB/xvfb-action@v1
  with:
    run: pytest tests/ -v --cov
  env:
    QT_QPA_PLATFORM: offscreen

- name: Run tests (macOS/Windows)
  run: pytest tests/ -v --cov
  env:
    QT_QPA_PLATFORM: offscreen

# Upload visual regression diffs on failure:
- name: Upload test artifacts
  if: failure()
  uses: actions/upload-artifact@v4
  with:
    name: test-screenshots-${{ matrix.os }}
    path: tests/baselines/*_diff.png
```

### Test Quality Assurance

After writing tests for each module, use the **`testreview-gemini`** skill to have Gemini review test coverage and confirm:
- All public methods have corresponding tests
- Edge cases are covered (empty inputs, boundary values, error paths)
- Mocks are used appropriately (not testing mocks instead of real code)
- Integration tests actually exercise the real code paths

### Test Dependencies (added to `[project.optional-dependencies]` in pyproject.toml)

```toml
[project.optional-dependencies]
test = [
    "pytest>=7.0",
    "pytest-qt>=4.2",
    "pytest-cov>=4.0",
    "scikit-image>=0.21",  # SSIM for visual regression
]
```

---

## Post-v0.1 Features (Steps 21-24)

All 20 original build steps are complete. The following features extend ClipShow:

### Step 21: Semantic Prompts + Pipeline Bug Fix (`clipshow-er5`)

**Bug Fix**: `pipeline.py:61` instantiated `SemanticDetector()` without passing `settings.semantic_prompts`. Fixed to pass `prompts=self.settings.semantic_prompts`.

**New Widget**: `clipshow/ui/prompt_editor.py` — Reusable `PromptEditor` with `QListWidget` (double-click to inline-edit), add/remove/reset buttons, duplicate/empty rejection, `prompts_changed(list)` signal.

**AnalyzePanel Update**: Two new detector rows (Semantic with "Edit Prompts..." button, Emotion) added after Motion, following the same checkbox+slider+label pattern. Prompts dialog uses `PromptEditor` embedded in a `QDialog`.

**Tests**: `tests/test_prompt_editor.py` (widget behavior), updates to `tests/test_pipeline.py` (custom prompts passed), updates to `tests/test_ui_analyze.py` (new slider rows).

### Step 22: Settings Dialog (`clipshow-21q`)

**New**: `clipshow/ui/settings_dialog.py` — Modal `QDialog` with grouped sections:
- Detector Weights (5x checkbox+slider)
- Semantic Prompts (embedded `PromptEditor`)
- Segment Selection (threshold slider, padding/duration spinboxes)
- Output Settings (codec combo, FPS spinbox, bitrate line edit)
- Save/Cancel/Reset buttons with snapshot-based cancel

**MainWindow Update**: `Edit > Preferences...` menu action opens the dialog. On save, syncs `analyze_panel._load_settings()`.

**Tests**: `tests/test_settings_dialog.py`, updates to `tests/test_ui_workflow.py`.

### Step 23: README (`clipshow-lns`)

User-facing `README.md` written for novices: elevator pitch, how detection works, prerequisites (Python 3.11+, FFmpeg), installation (`pip install -e .`, optional deps), usage (GUI walkthrough, auto mode, headless, CLI flags), customization (weights, thresholds, prompts), development setup, MIT license.

### Step 24: Linux Flatpak Packaging (`clipshow-1xe`)

**New files** in `packaging/`:
- `com.clipshow.app.yml` — Flatpak manifest using `org.kde.Platform//6.7`
- `com.clipshow.app.desktop` — XDG desktop entry
- `com.clipshow.app.metainfo.xml` — AppStream metadata

**Release workflow update**: New `build-linux` job in `.github/workflows/release.yml` — installs flatpak-builder, builds and bundles `.flatpak`, uploads as release artifact.

### Updated Project Structure (additions)

```
clipshow/
├── README.md                          # Step 23
├── clipshow/
│   └── ui/
│       ├── prompt_editor.py           # Step 21
│       └── settings_dialog.py         # Step 22
├── packaging/
│   ├── com.clipshow.app.yml           # Step 24
│   ├── com.clipshow.app.desktop       # Step 24
│   └── com.clipshow.app.metainfo.xml  # Step 24
└── tests/
    ├── test_prompt_editor.py          # Step 21
    └── test_settings_dialog.py        # Step 22
```
