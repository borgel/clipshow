"""Visual regression tests: screenshot-based UI verification.

Captures panel screenshots at 800x600 and compares against stored baselines
using scikit-image SSIM. Run with --update-baselines to regenerate baselines.
On failure, saves diff artifacts next to the baseline for debugging.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QSize
from PySide6.QtGui import QImage

from clipshow.config import Settings
from clipshow.model.moments import HighlightSegment
from clipshow.model.project import Project, VideoSource
from clipshow.ui.analyze_panel import AnalyzePanel
from clipshow.ui.export_panel import ExportPanel
from clipshow.ui.import_panel import ImportPanel
from clipshow.ui.main_window import MainWindow
from clipshow.ui.review_panel import ReviewPanel

BASELINES_DIR = Path(__file__).parent / "baselines"
CAPTURE_SIZE = QSize(800, 600)
SSIM_THRESHOLD = 0.95


def _qimage_to_array(img: QImage) -> np.ndarray:
    """Convert QImage to numpy array (RGB)."""
    img = img.convertToFormat(QImage.Format.Format_RGB888)
    w, h = img.width(), img.height()
    ptr = img.bits()
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, w, 3))
    return arr.copy()


def _capture_widget(widget, size: QSize = CAPTURE_SIZE) -> np.ndarray:
    """Grab a screenshot of a widget at the given size."""
    widget.resize(size)
    pixmap = widget.grab()
    return _qimage_to_array(pixmap.toImage())


def _save_image(arr: np.ndarray, path: Path) -> None:
    """Save a numpy array as PNG via QImage."""
    h, w, _ = arr.shape
    img = QImage(arr.data, w, h, w * 3, QImage.Format.Format_RGB888)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path))


def _load_image(path: Path) -> np.ndarray:
    """Load a PNG as numpy array."""
    img = QImage(str(path))
    return _qimage_to_array(img)


def _compute_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute SSIM between two images."""
    from skimage.metrics import structural_similarity

    return structural_similarity(a, b, channel_axis=2)


def _assert_visual_match(
    screenshot: np.ndarray,
    baseline_name: str,
    update: bool,
) -> None:
    """Compare screenshot against baseline or update it."""
    baseline_path = BASELINES_DIR / f"{baseline_name}.png"
    diff_path = BASELINES_DIR / f"{baseline_name}_diff.png"
    actual_path = BASELINES_DIR / f"{baseline_name}_actual.png"

    if update:
        _save_image(screenshot, baseline_path)
        # Clean up stale diff/actual artifacts
        diff_path.unlink(missing_ok=True)
        actual_path.unlink(missing_ok=True)
        return

    if not baseline_path.exists():
        pytest.skip(
            f"Baseline {baseline_path.name} not found. "
            "Run with --update-baselines to generate."
        )

    baseline = _load_image(baseline_path)

    if screenshot.shape != baseline.shape:
        _save_image(screenshot, actual_path)
        pytest.fail(
            f"Shape mismatch: {screenshot.shape} vs {baseline.shape}. "
            f"Actual saved to {actual_path}"
        )

    ssim = _compute_ssim(screenshot, baseline)
    if ssim < SSIM_THRESHOLD:
        _save_image(screenshot, actual_path)
        # Save diff image highlighting differences
        diff = np.abs(screenshot.astype(int) - baseline.astype(int)).astype(np.uint8)
        _save_image(diff, diff_path)
        pytest.fail(
            f"Visual regression: SSIM={ssim:.4f} < {SSIM_THRESHOLD}. "
            f"Diff saved to {diff_path}"
        )
    else:
        # Clean up old artifacts on pass
        diff_path.unlink(missing_ok=True)
        actual_path.unlink(missing_ok=True)


@pytest.fixture
def update_baselines(request):
    return request.config.getoption("--update-baselines")


def _make_sample_segments() -> list[HighlightSegment]:
    return [
        HighlightSegment(
            source_path="/tmp/video1.mp4",
            start_time=1.0,
            end_time=4.0,
            score=0.9,
            order=0,
        ),
        HighlightSegment(
            source_path="/tmp/video1.mp4",
            start_time=8.5,
            end_time=12.0,
            score=0.7,
            order=1,
        ),
        HighlightSegment(
            source_path="/tmp/video2.mp4",
            start_time=0.0,
            end_time=3.5,
            score=0.6,
            order=2,
        ),
    ]


class TestMainWindowVisual:
    def test_main_window_initial(self, qtbot, update_baselines):
        """Main window on first open — import tab, empty state."""
        window = MainWindow()
        qtbot.addWidget(window)
        screenshot = _capture_widget(window, CAPTURE_SIZE)
        _assert_visual_match(screenshot, "main_window_initial", update_baselines)

    def test_main_window_with_files(self, qtbot, update_baselines):
        """Main window with files loaded on import tab."""
        window = MainWindow()
        qtbot.addWidget(window)
        for i in range(3):
            source = VideoSource(
                path=f"/tmp/video{i}.mp4",
                duration=30.0 + i * 15,
                width=1920,
                height=1080,
            )
            window.import_panel.add_source_directly(source)
        screenshot = _capture_widget(window, CAPTURE_SIZE)
        _assert_visual_match(screenshot, "main_window_with_files", update_baselines)


class TestImportPanelVisual:
    def test_import_empty(self, qtbot, update_baselines):
        """Import panel — empty state."""
        project = Project()
        panel = ImportPanel(project)
        qtbot.addWidget(panel)
        screenshot = _capture_widget(panel, CAPTURE_SIZE)
        _assert_visual_match(screenshot, "import_empty", update_baselines)

    def test_import_loaded(self, qtbot, update_baselines):
        """Import panel — with files loaded."""
        project = Project()
        panel = ImportPanel(project)
        qtbot.addWidget(panel)
        for i in range(4):
            source = VideoSource(
                path=f"/home/user/videos/clip_{i:03d}.mp4",
                duration=10.0 + i * 5,
                width=1920,
                height=1080,
            )
            panel.add_source_directly(source)
        screenshot = _capture_widget(panel, CAPTURE_SIZE)
        _assert_visual_match(screenshot, "import_loaded", update_baselines)


class TestAnalyzePanelVisual:
    def test_analyze_defaults(self, qtbot, update_baselines):
        """Analyze panel — default settings."""
        project = Project()
        panel = AnalyzePanel(project)
        qtbot.addWidget(panel)
        screenshot = _capture_widget(panel, CAPTURE_SIZE)
        _assert_visual_match(screenshot, "analyze_defaults", update_baselines)

    def test_analyze_custom(self, qtbot, update_baselines):
        """Analyze panel — custom slider values."""
        project = Project()
        settings = Settings(
            scene_weight=0.8,
            audio_weight=0.3,
            motion_weight=0.6,
            score_threshold=0.4,
        )
        panel = AnalyzePanel(project, settings)
        qtbot.addWidget(panel)
        screenshot = _capture_widget(panel, CAPTURE_SIZE)
        _assert_visual_match(screenshot, "analyze_custom", update_baselines)


class TestReviewPanelVisual:
    def test_review_with_segments(self, qtbot, update_baselines):
        """Review panel — with segments loaded."""
        panel = ReviewPanel()
        qtbot.addWidget(panel)
        panel.set_segments(_make_sample_segments())
        screenshot = _capture_widget(panel, CAPTURE_SIZE)
        _assert_visual_match(screenshot, "review_with_segments", update_baselines)


class TestExportPanelVisual:
    def test_export_defaults(self, qtbot, update_baselines):
        """Export panel — default state."""
        panel = ExportPanel()
        qtbot.addWidget(panel)
        screenshot = _capture_widget(panel, CAPTURE_SIZE)
        _assert_visual_match(screenshot, "export_defaults", update_baselines)

    def test_export_with_segments(self, qtbot, update_baselines):
        """Export panel — with segments loaded showing summary."""
        panel = ExportPanel()
        qtbot.addWidget(panel)
        panel.set_segments(_make_sample_segments())
        screenshot = _capture_widget(panel, CAPTURE_SIZE)
        _assert_visual_match(screenshot, "export_with_segments", update_baselines)
