"""TUI application for interactive dataset labeling with vim-style keybindings."""

from enum import StrEnum
from typing import List, Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.reactive import reactive
from textual.widgets import Footer, Header, Label, Static

from budget.categories import Category
from budget.ml.active_learning.learner import ActiveLearner
from budget.ml.active_learning.models import Dataset, Record
from budget.ml.active_learning.strategies import Strategy


class RecordDisplay(Static):
    """Widget to display the current record data."""

    def __init__(self, record: Optional[Record] = None) -> None:
        super().__init__()
        self.record = record

    def update_record(self, record: Record) -> None:
        """Update the displayed record."""
        self.record = record
        self.update_display()

    def update_display(self) -> None:
        """Refresh the record display."""
        if not self.record:
            self.update("No record to display")
            return

        # Format record data in a readable way
        content = "📋 [bold]Current Record[/bold]\n\n"
        if isinstance(self.record.data, dict):
            for key, value in self.record.data.items():
                content += f"[bold]{key}[/bold]: {value}\n"
        else:
            content += f"[bold]Data[/bold]: {self.record.data}\n"

        if self.record.label:
            content += f"\n🏷️  [bold]Current Label[/bold]: {self.record.label}"
        else:
            content += "\n🏷️  [bold]Status[/bold]: Unlabeled"

        self.update(content)


class StatsPanel(Static):
    """Widget to display labeling statistics."""

    labeled_count: reactive[int] = reactive(0)
    unlabeled_count: reactive[int] = reactive(0)
    total_count: reactive[int] = reactive(0)

    def update_stats(self, dataset: Dataset) -> None:
        """Update statistics from dataset."""
        labeled = list(dataset.get_labeled())
        unlabeled = list(dataset.get_unlabeled())

        self.labeled_count = len(labeled)
        self.unlabeled_count = len(unlabeled)
        self.total_count = len(dataset.records)

    def watch_labeled_count(self) -> None:
        """React to changes in labeled count."""
        self.update_display()

    def watch_unlabeled_count(self) -> None:
        """React to changes in unlabeled count."""
        self.update_display()

    def update_display(self) -> None:
        """Update the stats display."""
        if self.total_count == 0:
            progress = 0
        else:
            progress = (self.labeled_count / self.total_count) * 100

        content = f"""📊 Progress

Labeled: {self.labeled_count}
Unlabeled: {self.unlabeled_count}
Total:** {self.total_count}
Progress: {progress:.1f}%"""

        self.update(content)


class LabelsPanel(Static):
    """Widget to display all available labels with selection highlighting."""

    selected_index: reactive[int] = reactive(0)

    def __init__(self, labels: List[str]) -> None:
        super().__init__()
        self.labels = labels

    def watch_selected_index(self) -> None:
        """React to changes in selected index."""
        self.update_display()

    def update_display(self) -> None:
        """Update the labels display with highlighting."""
        content = "🏷️ [bold]Labels[/bold]\n\n"

        for i, label in enumerate(self.labels):
            prefix = "▶ " if i == self.selected_index else "  "
            highlight = "[reverse]" if i == self.selected_index else ""
            end_highlight = "[/reverse]" if i == self.selected_index else ""

            content += f"{prefix}{highlight}{i + 1}. {label.upper()}{end_highlight}\n"

        self.update(content)

    def set_selected_index(self, index: int) -> None:
        """Set the selected label index."""
        if 0 <= index < len(self.labels):
            self.selected_index = index


class LabelingApp(App):
    """Main TUI application for dataset labeling."""

    CSS = """
    #main-container {
        layout: horizontal;
    }

    #record-panel {
        width: 2fr;
        padding: 1;
        border: solid $primary;
    }

    #control-panel {
        width: 1fr;
        padding: 1;
        border: solid $secondary;
    }

    #stats-panel {
        height: 8;
        border: solid $accent;
        margin-bottom: 1;
        padding: 1;
    }

    #labels-panel {
        border: solid $warning;
        padding: 1;
        margin-bottom: 1;
    }
    """

    BINDINGS = [
        # Vim-style navigation
        Binding("j", "next_label", "Next label", show=False),
        Binding("k", "prev_label", "Previous label", show=False),
        Binding("h", "prev_record", "Previous record", show=False),
        Binding("l", "next_record", "Next record", show=False),
        # Labeling actions
        Binding("enter", "confirm_label", "Apply label", show=True),
        Binding("s", "skip_record", "Skip record", show=True),
        # App controls
        Binding("q", "quit", "Quit", show=True),
        Binding("w", "save_dataset", "Save progress", show=True),
    ]

    def __init__(
        self,
        learner: ActiveLearner,
        labels: type[StrEnum],
        save_path: Optional[str] = None,
    ):
        super().__init__()
        self.learner = learner
        self.labels = [label.value for label in labels]
        self.save_path = save_path
        self.current_record: Optional[Record] = None
        self.selected_label_index = 0

    def compose(self) -> ComposeResult:
        """Create the TUI layout."""
        yield Header()

        with Container(id="main-container"):
            # Left panel - Record display
            with Vertical(id="record-panel"):
                yield Label("🏷️  Dataset Labeling Interface", classes="title")
                self.record_display = RecordDisplay()
                yield self.record_display

            # Right panel - Controls
            with Vertical(id="control-panel"):
                # Stats panel
                with Container(id="stats-panel"):
                    self.stats_panel = StatsPanel()
                    yield self.stats_panel

                # Labels panel
                with Container(id="labels-panel"):
                    self.labels_panel = LabelsPanel(self.labels)
                    yield self.labels_panel

        yield Footer()

    def on_mount(self) -> None:
        """Initialize the app when mounted."""
        self.load_next_record()
        self.labels_panel.update_display()
        self.stats_panel.update_stats(self.learner.dataset)

    def load_next_record(self) -> None:
        """Load the next unlabeled record."""
        try:
            self.current_record = self.learner.strategy.pick(self.learner.dataset)
            self.record_display.update_record(self.current_record)
        except Exception as e:
            self.notify(f"No more unlabeled records: {e}", severity="info")
            self.current_record = None
            self.record_display.update("🎉 All records have been processed!")

    def action_next_label(self) -> None:
        """Move to next label (vim j)."""
        self.selected_label_index = (self.selected_label_index + 1) % len(self.labels)
        self.labels_panel.set_selected_index(self.selected_label_index)

    def action_prev_label(self) -> None:
        """Move to previous label (vim k)."""
        self.selected_label_index = (self.selected_label_index - 1) % len(self.labels)
        self.labels_panel.set_selected_index(self.selected_label_index)

    def action_next_record(self) -> None:
        """Move to next record (vim l)."""
        self.action_skip_record()

    def action_prev_record(self) -> None:
        """Move to previous record (vim h) - for now, just skip."""
        self.notify("Previous record navigation not implemented", severity="warning")

    def action_confirm_label(self) -> None:
        """Apply the selected label to current record."""
        if not self.current_record:
            self.notify("No record to label", severity="warning")
            return

        selected_label = self.labels[self.selected_label_index]
        self.current_record.label_as(selected_label)
        self.stats_panel.update_stats(self.learner.dataset)
        self.load_next_record()

    def action_skip_record(self) -> None:
        """Skip current record without labeling."""
        self.load_next_record()

    def action_save_dataset(self) -> None:
        """Save current progress."""
        if self.save_path:
            try:
                self.learner.dataset.dump(self.save_path)
                self.notify(f"Progress saved to {self.save_path}", severity="success")
            except Exception as e:
                self.notify(f"Failed to save: {e}", severity="error")
        else:
            self.notify("No save path configured", severity="warning")


def launch_labeling_tui(
    dataset: Dataset,
    labels: type[StrEnum] = Category,
    strategy: Strategy | None = None,
    save_path: Optional[str] = None,
) -> None:
    """Launch the TUI labeling application.

    Args:
        dataset: The dataset to label
        labels: List of available labels
        strategy_name: Active learning strategy to use
        save_path: Optional path to save progress
    """
    from budget.ml.active_learning.strategies import RandomStrategy
    from budget.ml.active_learning.learner import ActiveLearner

    _strategy = strategy or RandomStrategy()

    learner = ActiveLearner(dataset, _strategy)

    app = LabelingApp(learner, labels, save_path)
    app.run()
