from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Protocol, Self

import pandas as pd
from pandas import DataFrame

LABEL_COLNAME = "__label__"


@dataclass
class Record:
    data: Any
    label: str | None = None

    def label_as(self, label: str) -> None:
        self.label = label


@dataclass
class Dataset:
    records: list[Record]

    def get_labeled(self) -> Iterator[Record]:
        return filter(lambda record: record.label is not None, self.records)

    def get_unlabeled(self) -> Iterator[Record]:
        return filter(lambda record: record.label is None, self.records)

    def to_dataframe(self) -> DataFrame:
        df = DataFrame.from_records([record.data for record in self.records])
        df = df.assign(**{LABEL_COLNAME: [record.label for record in self.records]})
        return df

    def dump(self, path: Path | str) -> None:
        df = self.to_dataframe()
        df.to_parquet(path)

    @classmethod
    def from_file(cls, path: Path) -> Self:
        df = pd.read_parquet(path)
        return cls.from_dataframe(df)

    @classmethod
    def from_dataframe(cls, df: DataFrame) -> Self:
        records = df.to_dict(orient="records")
        dataset = cls(
            records=[
                Record(data=record, label=record.pop(LABEL_COLNAME, None)) for record in records
            ]
        )
        return dataset


class Model(Protocol):
    def fit(self, X: Any, y: Any) -> Self: ...

    def predict(self, X: Any): ...
