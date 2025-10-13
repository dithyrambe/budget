from budget.exceptions import BudgetException


class NoMoreUnlabeledRecord(BudgetException):
    """Raises when trying to pick from a fully labeled dataset"""
