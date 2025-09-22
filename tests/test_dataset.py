from pathlib import Path

from budget.ml.active_learning.models import Dataset, Record


def test_dataset_dump_load(tmp_path: Path) -> None:
    data = [
        {"a": 1, "b": 3, "c": 5},
        {"a": 2, "b": 4, "c": 6},
    ]
    records = [Record(_) for _ in data]
    records[0].label_as("first")

    dataset = Dataset(records=records)
    filepath = tmp_path / "dataset.parquet"
    dataset.dump(filepath)
    loaded_dataset = Dataset.from_file(filepath)
    assert loaded_dataset.records[0].data == data[0]
    assert [record.label for record in loaded_dataset.records] == ["first", None]
