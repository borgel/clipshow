"""QApplication setup and auto-mode runner."""

import sys


def run_gui(files: list[str] | None = None) -> int:
    """Launch the full ClipShow GUI."""
    from PySide6.QtWidgets import QApplication

    from clipshow.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("ClipShow")
    app.setOrganizationName("ClipShow")

    window = MainWindow()
    if files:
        window.import_panel.add_files(files)
    window.show()

    return app.exec()


def run_auto_mode(
    files: list[str],
    output_path: str,
    headless: bool = False,
) -> int:
    """Run automatic highlight reel generation.

    Returns 0 on success, 1 on failure.
    """
    if not files:
        print("Error: no input files specified", file=sys.stderr)
        return 1

    # TODO: implement auto mode pipeline in Step 9
    print(f"Auto mode: {len(files)} files -> {output_path}")
    print("Auto mode not yet implemented")
    return 1
