"""Tests for the PromptEditor widget."""

import pytest

from clipshow.ui.prompt_editor import DEFAULT_NEGATIVE_PROMPTS, DEFAULT_PROMPTS, PromptEditor


@pytest.fixture()
def editor(qtbot):
    w = PromptEditor()
    qtbot.addWidget(w)
    return w


class TestInitialState:
    def test_default_prompts_loaded(self, editor):
        assert editor.prompts == DEFAULT_PROMPTS

    def test_list_count_matches(self, editor):
        assert editor.list_widget.count() == len(DEFAULT_PROMPTS)

    def test_remove_button_disabled(self, editor):
        assert editor.remove_button.isEnabled() is False

    def test_custom_initial_prompts(self, qtbot):
        custom = ["foo", "bar"]
        w = PromptEditor(prompts=custom)
        qtbot.addWidget(w)
        assert w.prompts == custom
        assert w.list_widget.count() == 2


class TestAdd:
    def test_add_new_prompt(self, editor, qtbot):
        editor.line_edit.setText("new prompt")
        with qtbot.waitSignal(editor.prompts_changed, timeout=1000):
            editor.add_button.click()
        assert "new prompt" in editor.prompts
        assert editor.list_widget.count() == len(DEFAULT_PROMPTS) + 1
        assert editor.line_edit.text() == ""

    def test_add_empty_rejected(self, editor):
        editor.line_edit.setText("   ")
        editor.add_button.click()
        assert editor.list_widget.count() == len(DEFAULT_PROMPTS)

    def test_add_duplicate_rejected(self, editor):
        editor.line_edit.setText(DEFAULT_PROMPTS[0])
        editor.add_button.click()
        assert editor.list_widget.count() == len(DEFAULT_PROMPTS)


class TestRemove:
    def test_remove_selected(self, editor, qtbot):
        editor.list_widget.setCurrentRow(0)
        removed = editor.prompts[0]
        with qtbot.waitSignal(editor.prompts_changed, timeout=1000):
            editor.remove_button.click()
        assert removed not in editor.prompts
        assert editor.list_widget.count() == len(DEFAULT_PROMPTS) - 1

    def test_remove_button_enabled_on_selection(self, editor):
        editor.list_widget.setCurrentRow(1)
        assert editor.remove_button.isEnabled() is True


class TestReset:
    def test_reset_restores_defaults(self, editor, qtbot):
        # Modify the list first
        editor.line_edit.setText("extra")
        editor.add_button.click()
        assert len(editor.prompts) == len(DEFAULT_PROMPTS) + 1

        with qtbot.waitSignal(editor.prompts_changed, timeout=1000):
            editor.reset_button.click()
        assert editor.prompts == DEFAULT_PROMPTS


class TestPerRowRemove:
    def test_row_remove_button_removes_prompt(self, editor, qtbot):
        original_count = len(editor.prompts)
        removed = editor.prompts[0]
        # Get the row widget and click its remove button
        item = editor.list_widget.item(0)
        row_widget = editor.list_widget.itemWidget(item)
        with qtbot.waitSignal(editor.prompts_changed, timeout=1000):
            row_widget.remove_clicked.emit(0)
        assert removed not in editor.prompts
        assert len(editor.prompts) == original_count - 1

    def test_each_row_has_remove_widget(self, editor):
        for i in range(editor.list_widget.count()):
            item = editor.list_widget.item(i)
            widget = editor.list_widget.itemWidget(item)
            assert widget is not None


class TestClearAll:
    def test_clear_all_empties_list(self, editor, qtbot):
        with qtbot.waitSignal(editor.prompts_changed, timeout=1000):
            editor.clear_all()
        assert editor.prompts == []
        assert editor.list_widget.count() == 0

    def test_clear_all_on_already_empty(self, qtbot):
        w = PromptEditor(prompts=[])
        qtbot.addWidget(w)
        w.clear_all()
        assert w.prompts == []

    def test_empty_prompts_not_replaced_by_defaults(self, qtbot):
        """Empty list should stay empty, not revert to defaults."""
        w = PromptEditor(prompts=[], default_prompts=["alpha", "beta"])
        qtbot.addWidget(w)
        assert w.prompts == []
        assert w.list_widget.count() == 0


class TestCustomDefaults:
    def test_custom_default_prompts(self, qtbot):
        custom_defaults = ["alpha", "beta"]
        w = PromptEditor(default_prompts=custom_defaults)
        qtbot.addWidget(w)
        assert w.prompts == custom_defaults

    def test_reset_uses_custom_defaults(self, qtbot):
        custom_defaults = ["alpha", "beta"]
        w = PromptEditor(prompts=["gamma"], default_prompts=custom_defaults)
        qtbot.addWidget(w)
        w.reset_button.click()
        assert w.prompts == custom_defaults

    def test_negative_prompts_defaults_exist(self):
        assert len(DEFAULT_NEGATIVE_PROMPTS) > 0

    def test_negative_editor_with_defaults(self, qtbot):
        w = PromptEditor(default_prompts=DEFAULT_NEGATIVE_PROMPTS)
        qtbot.addWidget(w)
        assert w.prompts == DEFAULT_NEGATIVE_PROMPTS
        w.reset_button.click()
        assert w.prompts == DEFAULT_NEGATIVE_PROMPTS
