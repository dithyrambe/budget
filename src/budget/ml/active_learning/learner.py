from enum import StrEnum
from typing import Optional

from budget.ml.active_learning.models import Dataset
from budget.ml.active_learning.strategies import Strategy


class ActiveLearner:
    def __init__(self, dataset: Dataset, strategy: Strategy) -> None:
        self.dataset = dataset
        self.strategy = strategy

    def prompt_for_label(self) -> None:
        record = self.strategy.pick(self.dataset)
        print(record)
        label = input("Input label:")
        record.label_as(label)

    def launch_tui(self, labels: type[StrEnum], save_path: Optional[str] = None) -> None:
        from budget.ml.active_learning.tui import launch_labeling_tui

        launch_labeling_tui(self.dataset, labels=labels, save_path=save_path)

    def set_strategy(self, strategy: Strategy) -> None:
        self.strategy = strategy
