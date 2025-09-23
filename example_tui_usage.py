#!/usr/bin/env python3
"""Example usage of the TUI labeling application."""

from budget.categories import Category
from budget.transaction_loader import BanquePopulaireLoader
from budget.ml.active_learning.learner import ActiveLearner
from budget.ml.active_learning.models import Dataset
from budget.ml.active_learning.strategies import RandomStrategy


def main():
    loader = BanquePopulaireLoader()
    transactions = loader.read("/home/lvpasquier/Téléchargements/22092025_456389.csv")
    dataset = Dataset.from_dataframe(transactions)

    strategy = RandomStrategy()
    learner = ActiveLearner(dataset, strategy)
    learner.launch_tui(labels=Category, save_path="/tmp/yolo.parquet")

    print("✅ Labeling session completed!")

    labeled = list(dataset.get_labeled())
    unlabeled = list(dataset.get_unlabeled())

    print("\n📊 Final Statistics:")
    print(f"   Labeled: {len(labeled)}")
    print(f"   Unlabeled: {len(unlabeled)}")
    print(f"   Total: {len(dataset.records)}")

    if labeled:
        print("\n🏷️  Sample labeled records:")
        for record in labeled[:3]:
            print(f"   {record.data['description'][:40]:<40} → {record.label}")


if __name__ == "__main__":
    main()
