# ClipShow Project Memory

## User Preferences

- **Use `uv` for all Python commands** — always `uv run pytest`, `uv run ruff`, etc. Never `source .venv/bin/activate`
- **Do NOT create PRs** — just push branches directly, user handles PRs themselves
- **Use Gemini** (`testreview-gemini` / `testaudit-gemini`) when unsure about test quality or issue root cause
- **Commit work as it is completed** — do not accumulate uncommitted changes across multiple tasks
- Use `bd` (beads) to track ALL work — always check before starting, always update
- For Qt UI tests, use `button.click()` not `qtbot.mouseClick()` (PySide6 signature issue)

## CRITICAL: Work Tracking with Beads

1. **Before starting any work**, run `bd list` to see current beads
2. **Pick work from beads** — do not freelance
3. **Update bead status** when starting: `bd update <id> --status=in_progress`
4. **Close beads when done**: `bd close <id>` after work committed
5. **Create new beads** for discovered bugs/tasks
6. **Include beads files in feature commits** — stage `.beads/issues.jsonl` alongside code

## CRITICAL: Release Verification Checklist

**Every release MUST pass these smoke tests before being considered done:**

1. **Linux CI smoke test** — runs automatically as `smoke-test-linux` job in `release.yml`
   - Installs Flatpak, runs `--headless` against test video, checks exit code AND stderr
2. **macOS local smoke test** — run manually on the dev machine:
   ```
   ./scripts/smoke-test-macos.sh <TAG> lite
   ./scripts/smoke-test-macos.sh <TAG> full
   ```
   - Downloads DMG, mounts, runs `--headless`, checks exit code, stderr, and output file
3. **Verify artifact sizes** — full builds should be ~300-400MB larger than lite
4. **Check `gh release view <TAG>`** — all 5 artifacts present

## Project Structure

- **Tech stack**: PySide6, MoviePy 2.x, PySceneDetect, librosa, OpenCV, ONNX Runtime
- **Testing**: pytest + pytest-qt, headless with `QT_QPA_PLATFORM=offscreen`
- **Packaging**: PyInstaller (macOS/Windows), Flatpak (Linux), lite + full variants
- **CI**: `.github/workflows/ci.yml` (tests), `.github/workflows/release.yml` (builds + smoke test)
- **CLI**: `clipshow --auto --headless --output out.mp4 input.mp4` for headless processing

## Key Packaging Details

- Both `.spec` files must include: `collect_data_files("cv2", includes=["**/*.xml"])` for OpenCV haar cascades
- Both `.spec` files must include: `copy_metadata("imageio")` for moviepy compatibility
- Flatpak manifest uses pinned requirements (`uv export`) + `--only-binary numpy,scipy,scikit-learn`
- Flatpak bundles static ffmpeg/ffprobe from BtbN builds (downloaded in CI)
- Version is read dynamically from `pyproject.toml` in both spec files (use `SPECPATH`, not `__file__`)
- Windows installer version passed via `/DAppVer=` to Inno Setup

## Known Patterns

- `--only-binary :all:` breaks Flatpak builds (prevents building local package from source)
- Emotion detector downloads model atomically (tempfile + os.replace) to avoid race conditions
- Pipeline catches `Exception` (not just `RuntimeError`) for optional detector failures
- Flaky audio fixture tests are skipped in CI: `-m "not uses_audio_fixture"`
