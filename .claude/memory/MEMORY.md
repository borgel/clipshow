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

- **Branch**: `feature/step2-ffprobe` (contains Steps 1-8 and 10)
- **All 90 tests passing** — run with `QT_QPA_PLATFORM=offscreen uv run pytest tests/ -v`
- **Lint clean** — `uv run ruff check clipshow/ tests/`
- **No remote configured** — no `git push` possible yet

### Completed Steps (9 of 20)
- Step 1: Project scaffold (pyproject.toml, models, config, CI, conftest fixtures)
- Step 2: `export/ffprobe.py` — metadata extraction via ffprobe subprocess
- Step 3: `detection/scoring.py` — weighted combination, thresholding, segment extraction
- Step 4: `detection/scene.py` — PySceneDetect ContentDetector wrapper
- Step 5: `detection/audio.py` — librosa onset strength + RMS energy
- Step 6: `detection/motion.py` — OpenCV frame differencing
- Step 7: `detection/pipeline.py` — orchestrates all detectors + scoring
- Step 8: `export/assembler.py` — MoviePy clip concatenation
- Step 10: `ui/main_window.py` — 4-tab workflow shell with navigation

### Next Unblocked Steps (pick from these)
- **Step 9** (`clipshow-8zn`): Auto Mode — wire `run_auto_mode()` in app.py, full CLI pipeline
- **Step 11** (`clipshow-uzs`): Import Panel — drag-drop file import UI + test_ui_import.py
- **Step 12** (`clipshow-7sh`): Video Preview — QMediaPlayer wrapper
- **Step 17** (`clipshow-8v0`): Semantic Detector (optional, standalone)
- **Step 18** (`clipshow-5si`): Emotion Detector (optional, standalone)

### Critical Path
Steps 11+12 → Step 13 (Analyze Panel) → Step 14 (Review) → Step 15 (Export) → Step 16 (E2E)

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
