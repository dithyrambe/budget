from enum import StrEnum
from typing import Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

from budget.ml.active_learning.models import Dataset
from budget.ml.active_learning.strategies import Strategy


class EventDateEncoder(BaseEstimator, TransformerMixin):
    """Normalize event_date to float between 0 (start of month) and 1 (end of month)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X.iloc[0], str):
            dates = pd.to_datetime(X)
        else:
            dates = X

        days_of_month = dates.dt.day
        days_in_month = dates.dt.days_in_month
        normalized = (days_of_month - 1) / (days_in_month - 1)

        return normalized.values.reshape(-1, 1)


class ActiveLearner:
    def __init__(self, dataset: Dataset, strategy: Strategy) -> None:
        self.dataset = dataset
        self.strategy = strategy

    def prompt_for_label(self) -> None:
        pick = self.strategy.pick(self.dataset)
        print(pick)
        label = input("Input label:")
        pick.record.label_as(label)

    def launch_tui(
        self,
        labels: type[StrEnum],
        save_path: Optional[str] = None,
    ) -> None:
        from budget.ml.active_learning.tui import launch_labeling_tui

        launch_labeling_tui(
            self.dataset,
            labels=labels,
            strategy=self.strategy,
            save_path=save_path,
        )

    def set_strategy(self, strategy: Strategy) -> None:
        self.strategy = strategy


def get_default_model():
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "event_date",
                EventDateEncoder(),
                "event_date",
            ),
            (
                "description_vec",
                CountVectorizer(
                    max_features=3000,
                    lowercase=True,
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                ),
                "description",
            ),
            (
                "category_vec",
                CountVectorizer(
                    max_features=1000,
                    lowercase=True,
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                ),
                "category",
            ),
            (
                "subcategory_vec",
                CountVectorizer(
                    max_features=1000,
                    lowercase=True,
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                ),
                "subcategory",
            ),
            (
                "amount",
                "passthrough",
                ["amount"],
            ),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                LGBMClassifier(
                    random_state=42,
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    verbose=-1,
                ),
            ),
        ]
    )

    return pipeline
