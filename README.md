# ClipShow

ClipShow is a desktop app that turns your raw video clips into a highlight reel automatically. Point it at your footage, and it finds the most interesting moments using scene changes, audio peaks, motion, and optional AI-powered content analysis, then stitches them together into a single video.

## How It Works

ClipShow runs several **detectors** on your video files:

- **Scene change** — spots hard cuts and transitions between shots
- **Audio peaks** — finds loud moments, speech, beats, and cheers
- **Motion** — detects fast-moving action on screen
- **Semantic (optional)** — uses a CLIP AI model to find frames matching text descriptions you choose (e.g. "crashes", "celebrations", "beautiful scenery")
- **Emotion (optional)** — detects faces and flags moments with happy or surprised expressions

Each detector produces a score over time. ClipShow combines these scores using weights you control, then picks segments that rise above a threshold. You can preview, reorder, and trim those segments before exporting the final highlight reel.

## Prerequisites

- **Python 3.11 or 3.12**
- **FFmpeg** — must be installed and available in your system PATH

### Installing FFmpeg

| OS | Command |
|----|---------|
| macOS | `brew install ffmpeg` |
| Windows | `winget install ffmpeg` or `choco install ffmpeg -y` |
| Ubuntu/Debian | `sudo apt install ffmpeg` |
| Fedora | `sudo dnf install ffmpeg` |

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/borgel/clipshow.git
cd clipshow
pip install -e .
```

### Optional dependencies

For AI-powered semantic detection (CLIP model, ~340 MB download on first use):

```bash
pip install -e ".[semantic]"
```

For face/emotion detection:

```bash
pip install -e ".[emotion]"
```

## Usage

### GUI mode (default)

```bash
clipshow
```

This opens the four-step workflow:

1. **Import** — drag and drop video files or use the file browser
2. **Analyze** — adjust detector weights and threshold, then click "Analyze All"
3. **Review** — preview detected segments, reorder by dragging, trim with nudge buttons, uncheck segments you don't want
4. **Export** — pick an output path and click "Export"

You can also open the **Edit > Preferences** menu to change all settings in one place.

### Auto mode

Process videos and produce a highlight reel with one command:

```bash
clipshow --auto --output reel.mp4 video1.mp4 video2.mov
```

A minimal progress window is shown. All segments above the threshold are included, sorted chronologically.

### Headless mode

For scripting or batch jobs — no GUI at all:

```bash
clipshow --auto --headless --output reel.mp4 *.mp4
```

### CLI flags

| Flag | Description |
|------|-------------|
| `--auto` | Run automatic mode (no full UI) |
| `--headless` | Suppress all GUI (implies `--auto`) |
| `--output`, `-o` | Output file path (default: `highlight_reel.mp4`) |
| `--version` | Print version and exit |

## Customizing Detection

All settings are stored in `~/.clipshow/settings.json` and can be edited in the GUI via **Edit > Preferences**.

### Detector weights

Each detector has a weight from 0.0 (disabled) to 1.0 (full strength). Defaults:

| Detector | Default weight |
|----------|---------------|
| Scene | 0.30 |
| Audio | 0.25 |
| Motion | 0.25 |
| Semantic | 0.00 (disabled) |
| Emotion | 0.20 |

Set a weight to 0 to disable a detector entirely.

### Threshold and duration

- **Score threshold** (default 0.5) — only moments scoring above this are kept. Lower it to get more clips, raise it for only the best moments.
- **Pre-padding** (default 1.0 s) — seconds of video to include before each detected moment
- **Post-padding** (default 1.5 s) — seconds to include after
- **Min segment duration** (default 1.0 s) — discard segments shorter than this
- **Max segment duration** (default 15.0 s) — cap segment length

### Custom semantic prompts

When using the semantic detector, ClipShow scores video frames against text descriptions. The defaults are "exciting moment", "people laughing", and "beautiful scenery".

You can customize these in the Analyze panel (click "Edit Prompts...") or in Preferences. Examples of useful prompts:

- `"car crashes"` — find collision moments in dashcam footage
- `"goal celebration"` — find sports highlights
- `"sunset over water"` — find scenic landscape shots
- `"crowd cheering"` — find audience reaction moments

## Development

### Setup

```bash
git clone https://github.com/borgel/clipshow.git
cd clipshow
pip install -e ".[test]"
```

### Running tests

```bash
uv run pytest tests/ -v
```

Tests include unit tests, integration tests with synthetic videos, UI tests (runs headlessly via `QT_QPA_PLATFORM=offscreen`), and visual regression tests.

To update visual regression baselines after UI changes:

```bash
uv run pytest tests/test_visual_regression.py --update-baselines
```

### Linting

```bash
uv run ruff check clipshow/ tests/
```

## License

MIT — see [LICENSE](LICENSE) for details.
