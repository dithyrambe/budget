from pathlib import Path

import pytest

from budget.transaction_loader.banque_populaire import BanquePopulaireLoader
from budget.transaction_loader.credit_lyonnais import CreditLyonnaisLoader


@pytest.mark.parametrize(
    argnames=("loader", "path"),
    argvalues=[
        (BanquePopulaireLoader(), "banque_populaire.csv"),
        (CreditLyonnaisLoader(), "credit_lyonnais.csv")
    ],
)
def test_banque_populaire_loader(loader, path):
    path = Path(__file__).parent / "data" / "transactions" / path
    df = loader.read(path)
    assert not df.empty
