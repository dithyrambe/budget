"""Labeling CLI"""

from enum import Enum
from typing import Optional
from typer import Exit, Typer, Option, echo

import pandas as pd

from budget.categories import Category
from budget.transaction_loader import BanquePopulaireLoader
from budget.ml.active_learning.learner import ActiveLearner, get_default_model
from budget.ml.active_learning.models import LABEL_COLNAME, Dataset
from budget.ml.active_learning.strategies import AmbiguousStrategy
from budget.transaction_loader.base import TransactionLoader


cli = Typer(add_completion=False)


class Loader(Enum):
    BANQUE_POPULAIRE = "banque-populaire"

    @property
    def loader(self) -> TransactionLoader:
        loaders = {
            Loader.BANQUE_POPULAIRE.value: BanquePopulaireLoader(),
        }
        return loaders[self.value]


@cli.command()
def launch_ui(
    loader: Optional[Loader] = Option(
        None,
        "-l",
        "--loader",
        help="Transaction loader",
    ),
    _from: Optional[str] = Option(
        None,
        "-f",
        "--from",
        help="Transaction file to load (ignored if --resume-from)",
    ),
    resume_from: Optional[str] = Option(
        None,
        help="Resume labeling from dump",
    ),
    output: str = Option(
        ...,
        "-o",
        "--output",
        help="Location to dump checkpoint",
    ),
):
    if resume_from:
        transactions = pd.read_parquet(resume_from)
    else:
        if not _from or not loader:
            echo(
                "You need to specify a transaction file (--from) "
                "and a loader (--loader) to start a labeling session",
                err=True,
            )
            raise Exit(1)
        transactions = loader.loader.read(_from)

    dataset = Dataset.from_dataframe(transactions)
    training_dataset = Dataset(records=[*dataset.get_labeled()])
    training_tx = training_dataset.to_dataframe()

    model = get_default_model()
    model.fit(X=training_tx, y=training_tx[LABEL_COLNAME])
    __import__("IPython").embed()

    strategy = AmbiguousStrategy(model=model, refit=True)
    learner = ActiveLearner(dataset=dataset, strategy=strategy)
    learner.launch_tui(labels=Category, save_path=output)

    echo("✅ Labeling session completed!")

    labeled = list(dataset.get_labeled())
    unlabeled = list(dataset.get_unlabeled())

    print("\n📊 Statistics:")
    print(f"   Labeled: {len(labeled)}")
    print(f"   Unlabeled: {len(unlabeled)}")
    print(f"   Total: {len(dataset.records)}")
