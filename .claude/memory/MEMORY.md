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

## Project Structure

- **Plan**: `PLAN.md` in repo root — full architecture and build order
- **Build order**: 20 steps, tracked as beads with dependency chains
- **Git workflow**: Feature branches per step, merge to main
- **Tech stack**: PySide6, MoviePy 2.x, PySceneDetect, librosa, OpenCV, ONNX Runtime
- **Testing**: pytest + pytest-qt, headless with QT_QPA_PLATFORM=offscreen

## User Preferences

- **Use `uv` for all Python commands** — always `uv run pytest`, `uv run ruff`, etc. Never `source .venv/bin/activate`
- Use `bd` (beads) to track ALL work — always check before starting, always update
- Modular git commits in feature branches
- Run `testreview-gemini` after writing tests for each module
- Keep memory files in `.claude/memory/` in the repo (tracked in git)
