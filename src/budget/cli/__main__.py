from typer import Exit, Option, Typer, echo

from budget import PACKAGE_NAME


cli = Typer(add_completion=False)


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
