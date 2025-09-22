from abc import ABC, abstractmethod
from random import choice

from budget.ml.active_learning.exceptions import NoMoreUnlabeledRecord
from budget.ml.active_learning.models import Dataset, Record


class Strategy(ABC):
    @abstractmethod
    def pick(self, dataset: Dataset) -> Record: ...


class RandomStrategy(Strategy):
    def pick(self, dataset: Dataset) -> Record:
        unlabeled = [*dataset.get_unlabeled()]
        if not unlabeled:
            raise NoMoreUnlabeledRecord("All dataset records has been labeled")

        return choice(unlabeled)
