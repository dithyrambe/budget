from typer import Exit, Option, Typer, echo

from budget import PACKAGE_NAME
from budget.cli.labeling import cli as labeling_cli


cli = Typer(add_completion=False)
cli.add_typer(labeling_cli, name="labeling")


def show_version(flag: bool):
    if flag:
        from importlib.metadata import version

        echo(f"{PACKAGE_NAME} {version(PACKAGE_NAME)}")
        raise Exit()


@cli.callback()
def main(
    version: bool = Option(None, "--version", callback=show_version, help="Show version."),
):
    """Pyre CLI"""


if __name__ == "__main__":
    cli()
