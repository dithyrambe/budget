from pathlib import Path

import pandas as pd
import pendulum

from budget.transaction_loader import TransactionLoader


class CreditLyonnaisLoader(TransactionLoader):
    def __init__(
        self,
        sep: str = ";",
        decimal: str = ",",
        encoding: str = "latin",
        strict: bool = True,
    ):
        TransactionLoader.__init__(
            self, sep=sep, decimal=decimal, encoding=encoding, strict=strict
        )

    def read_raw(self, path: str | Path) -> pd.DataFrame:
        columns = [
            "Date",
            "Montant",
            "Type",
            "Compte",
            "Desc. debit",
            "Desc. credit",
            "Carte",
            "Categorie",
        ]
        return pd.read_csv(
            path,
            sep=self.sep,
            decimal=self.decimal,
            encoding=self.encoding,
            names=columns,
            skiprows=1,
            skipfooter=1,
            engine="python",
        )

    def get_event_date(self, df: pd.DataFrame) -> pd.Series:
        return df["Date"].apply(
            lambda date: pendulum.from_format(date, fmt="DD/MM/YYYY").to_date_string()
        )

    def get_event_datetime(self, df: pd.DataFrame) -> pd.Series:
        return df["Date"].apply(
            lambda date: pendulum.from_format(
                date, fmt="DD/MM/YYYY"
            ).to_iso8601_string()
        )

    def get_description(self, df: pd.DataFrame) -> pd.Series:
        return df["Desc. debit"].fillna(df["Desc. credit"])

    def get_amount(self, df: pd.DataFrame) -> pd.Series:
        return df["Montant"].fillna(0.0)

    def get_category(self, df: pd.DataFrame) -> pd.Series:
        return df["Categorie"].fillna("Categorie")

    def get_subcategory(self, df: pd.DataFrame) -> pd.Series:
        return df["Categorie"].fillna("Sous-categorie")
