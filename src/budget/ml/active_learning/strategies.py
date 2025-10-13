from abc import ABC, abstractmethod
from dataclasses import dataclass
from random import choice
from typing import Any

from numpy.typing import NDArray
import numpy as np

from budget.ml.active_learning.exceptions import NoMoreUnlabeledRecord
from budget.ml.active_learning.models import LABEL_COLNAME, Dataset, Record


@dataclass
class Pick:
    record: Record
    scores: list[float] | None = None


class Strategy(ABC):
    @abstractmethod
    def pick(self, dataset: Dataset) -> Pick: ...


class RandomStrategy(Strategy):
    def pick(self, dataset: Dataset) -> Pick:
        unlabeled = [*dataset.get_unlabeled()]
        if not unlabeled:
            raise NoMoreUnlabeledRecord("All dataset records has been labeled")

        return Pick(record=choice(unlabeled))


class AmbiguousStrategy(Strategy):
    def __init__(self, model: Any, refit: bool = False) -> None:
        self.model = model
        self.refit = refit

    def pick(self, dataset: Dataset) -> Pick:
        if self.refit:
            training_dataset = Dataset(records=[*dataset.get_labeled()])
            training_tx = training_dataset.to_dataframe()
            self.model.fit(X=training_tx, y=training_tx[LABEL_COLNAME])

        unlabeled = Dataset(records=[*dataset.get_unlabeled()])
        if not unlabeled:
            raise NoMoreUnlabeledRecord("All dataset records has been labeled")
        X = unlabeled.to_dataframe().drop(LABEL_COLNAME, axis="columns")
        preds = self.model.predict_proba(X)
        gap = self.get_best_candidates_gap(preds)
        most_ambiguous = gap.argmin()
        return Pick(
            record=unlabeled.records[most_ambiguous],
            scores=preds[most_ambiguous].tolist(),
        )

    def get_best_candidates_gap(self, predictions: NDArray) -> NDArray:
        top2_indices = np.argsort(predictions, axis=-1)[:, -2:]
        top2 = np.take_along_axis(predictions, top2_indices, axis=-1)
        gap = np.squeeze(np.diff(top2, axis=-1))
        return gap
