"""Reusable list-editor widget for semantic prompts."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

DEFAULT_PROMPTS = [
    "exciting moment",
    "people laughing",
    "beautiful scenery",
]


class PromptEditor(QWidget):
    """Editable list of semantic text prompts.

    Emits ``prompts_changed`` whenever the list is modified (add, remove,
    inline edit, or reset).
    """

    prompts_changed = Signal(list)

    def __init__(
        self,
        prompts: list[str] | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._prompts: list[str] = list(prompts or DEFAULT_PROMPTS)
        self._setup_ui()
        self._connect_signals()
        self._refresh_list()

    # ── UI ────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget)

        # Add row
        add_row = QHBoxLayout()
        self.line_edit = QLineEdit()
        self.line_edit.setPlaceholderText("New prompt…")
        self.add_button = QPushButton("Add")
        add_row.addWidget(self.line_edit)
        add_row.addWidget(self.add_button)
        layout.addLayout(add_row)

        # Action buttons
        btn_row = QHBoxLayout()
        self.remove_button = QPushButton("Remove")
        self.remove_button.setEnabled(False)
        self.reset_button = QPushButton("Reset to Defaults")
        btn_row.addWidget(self.remove_button)
        btn_row.addStretch()
        btn_row.addWidget(self.reset_button)
        layout.addLayout(btn_row)

    def _connect_signals(self) -> None:
        self.add_button.clicked.connect(self._on_add)
        self.line_edit.returnPressed.connect(self._on_add)
        self.remove_button.clicked.connect(self._on_remove)
        self.reset_button.clicked.connect(self._on_reset)
        self.list_widget.currentRowChanged.connect(self._on_selection_changed)
        self.list_widget.itemChanged.connect(self._on_item_edited)

    # ── Slots ─────────────────────────────────────────────────────────

    def _on_add(self) -> None:
        text = self.line_edit.text().strip()
        if not text:
            return
        if text in self._prompts:
            return
        self._prompts.append(text)
        self.line_edit.clear()
        self._refresh_list()
        self.prompts_changed.emit(list(self._prompts))

    def _on_remove(self) -> None:
        row = self.list_widget.currentRow()
        if 0 <= row < len(self._prompts):
            self._prompts.pop(row)
            self._refresh_list()
            self.prompts_changed.emit(list(self._prompts))

    def _on_reset(self) -> None:
        self._prompts = list(DEFAULT_PROMPTS)
        self._refresh_list()
        self.prompts_changed.emit(list(self._prompts))

    def _on_selection_changed(self, row: int) -> None:
        self.remove_button.setEnabled(row >= 0)

    def _on_item_edited(self, item: QListWidgetItem) -> None:
        row = self.list_widget.row(item)
        new_text = item.text().strip()

        # Reject empty or duplicate edits
        if not new_text or (new_text != self._prompts[row] and new_text in self._prompts):
            self.list_widget.blockSignals(True)
            item.setText(self._prompts[row])
            self.list_widget.blockSignals(False)
            return

        if new_text != self._prompts[row]:
            self._prompts[row] = new_text
            self.prompts_changed.emit(list(self._prompts))

    # ── Helpers ───────────────────────────────────────────────────────

    def _refresh_list(self) -> None:
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        for prompt in self._prompts:
            item = QListWidgetItem(prompt)
            item.setFlags(item.flags() | item.flags().ItemIsEditable)
            self.list_widget.addItem(item)
        self.list_widget.blockSignals(False)

    @property
    def prompts(self) -> list[str]:
        return list(self._prompts)

    @prompts.setter
    def prompts(self, value: list[str]) -> None:
        self._prompts = list(value)
        self._refresh_list()
