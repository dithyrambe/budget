from budget.ml.active_learning.models import Dataset, Model
from budget.ml.active_learning.strategies import Strategy


class ActiveLearner:
    def __init__(self, model: Model, dataset: Dataset, strategy: Strategy) -> None:
        self.model = model
        self.dataset = dataset
        self.strategy = strategy

    def prompt_for_label(self) -> None:
        record = self.strategy.pick(self.dataset)
        print(record)
        label = input("Input label:")
        record.label_as(label)

    def set_strategy(self, strategy: Strategy) -> None:
        self.strategy = strategy
