# ClipShow Project Memory

## CRITICAL: Work Tracking with Beads

**ALL agents working on this project MUST follow these rules:**

1. **Before starting any work**, run `bd list` to see current beads (issues) and their status
2. **Pick work from beads** — do not freelance. Find an open, unblocked bead to work on
3. **Update bead status** when starting work: `bd update <id> --assignee <name>`
4. **Close beads when done**: `bd close <id>` after work is complete and committed
5. **Create new beads** for any discovered bugs or tasks: `bd create "title" -d "description"`
6. **Use feature branches** — one branch per bead/step, e.g. `feature/step1-scaffold`
7. **Commit modularly** — each commit should be atomic and relate to the bead being worked

### Quick Reference
- `bd list` — see all open issues
- `bd show <id>` — see full details of an issue
- `bd close <id>` — mark done
- `bd create "title" -d "desc"` — create new issue
- `bd dep tree <id>` — see dependency tree

## Current State

- **Branch**: `feature/step11-import-panel` (contains Steps 1-8, 10-16)
- **233 tests passing** (2 pre-existing scene detector failures)
- **Lint clean** — `uv run ruff check clipshow/ tests/`
- **No remote configured** — no `git push` possible yet

### Completed Steps (15 of 20)
- Step 1: Project scaffold (pyproject.toml, models, config, CI, conftest fixtures)
- Step 2: `export/ffprobe.py` — metadata extraction via ffprobe subprocess
- Step 3: `detection/scoring.py` — weighted combination, thresholding, segment extraction
- Step 4: `detection/scene.py` — PySceneDetect ContentDetector wrapper
- Step 5: `detection/audio.py` — librosa onset strength + RMS energy
- Step 6: `detection/motion.py` — OpenCV frame differencing
- Step 7: `detection/pipeline.py` — orchestrates all detectors + scoring
- Step 8: `export/assembler.py` — MoviePy clip concatenation
- Step 10: `ui/main_window.py` — 4-tab workflow shell, wired to all panels
- Step 11: `ui/import_panel.py` — drag-drop, file list, browse, remove/clear
- Step 12: `ui/video_preview.py` — QMediaPlayer wrapper with segment playback
- Step 13: `ui/analyze_panel.py` + `workers/analysis_worker.py` — detector settings + QThread
- Step 14: `ui/review_panel.py` + `ui/segment_list.py` — segment list, trim, preview
- Step 15: `ui/export_panel.py` + `workers/export_worker.py` — output settings + QThread
- Step 16: `test_ui_workflow.py` — full E2E UI journey

### Remaining Steps (pick from these)
- **Step 9** (`clipshow-8zn`): Auto Mode — wire `run_auto_mode()` in app.py, full CLI pipeline
- **Step 17** (`clipshow-8v0`): Semantic Detector (optional, standalone)
- **Step 18** (`clipshow-5si`): Emotion Detector (optional, standalone)
- **Step 19** (`clipshow-9ck`): Visual Regression Baselines
- **Step 20** (`clipshow-di1`): Packaging & CI/CD

### Known Issues
- 2 pre-existing scene detector test failures (PySceneDetect not detecting scene change in synthetic video)
- Qt `.click()` on lambda-connected buttons may not fire in offscreen mode — use direct method calls in tests

## Project Structure

- **Plan**: `PLAN.md` in repo root — full architecture and build order
- **Build order**: 20 steps, tracked as beads with dependency chains
- **Git workflow**: Feature branches per step, merge to main
- **Tech stack**: PySide6, MoviePy 2.x, PySceneDetect, librosa, OpenCV, ONNX Runtime
- **Testing**: pytest + pytest-qt, headless with QT_QPA_PLATFORM=offscreen
- **Skills**: `testaudit-gemini` in `.claude/skills/` — thorough Gemini-based test audit

## User Preferences

- **Use `uv` for all Python commands** — always `uv run pytest`, `uv run ruff`, etc. Never `source .venv/bin/activate`
- Use `bd` (beads) to track ALL work — always check before starting, always update
- Modular git commits in feature branches
- Run `testreview-gemini` or `testaudit-gemini` after writing tests for each module
- Keep memory files in `.claude/memory/` in the repo (tracked in git)
- For Qt UI tests, use `button.click()` not `qtbot.mouseClick()` (PySide6 signature issue)
