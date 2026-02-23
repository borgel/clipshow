"""Qt signals for cross-component updates."""

from PySide6.QtCore import QObject, Signal


class ProjectSignals(QObject):
    """Signals emitted when project state changes."""

    sources_changed = Signal()
    analysis_started = Signal()
    analysis_progress = Signal(str, float)  # source_path, progress 0-1
    analysis_complete = Signal()
    segments_changed = Signal()
    export_started = Signal()
    export_progress = Signal(float)  # progress 0-1
    export_complete = Signal(str)  # output_path
    export_error = Signal(str)  # error message
