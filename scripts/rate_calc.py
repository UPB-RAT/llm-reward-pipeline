import typer
from pathlib import Path
from typing import Optional
from rich.table import Table
from rich.console import Console

console = Console()


def _scan(rewards_dir: Path) -> tuple[int, int]:
    rejected = sum(1 for f in rewards_dir.iterdir() if f.is_file() and f.name.startswith("rejected"))
    reward = sum(1 for f in rewards_dir.iterdir() if f.is_file() and f.name.startswith("reward"))
    return reward, rejected


def _print_table(title: str, reward: int, rejected: int):
    total = reward + rejected
    rate = (reward / total * 100) if total > 0 else 0.0
    table = Table(title=title)
    table.add_column("Category", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Percentage", justify="right")
    table.add_row("Reward", str(reward), f"{rate:.2f}%")
    table.add_row("Rejected", str(rejected), f"{100 - rate:.2f}%")
    table.add_row("Total", str(total), "100.00%")
    console.print(table)
    console.print()


def main(
    output_dirs: Optional[list[Path]] = typer.Argument(
        None,
        help="One or more output directories containing a rewards/ subdirectory. "
        "If omitted, auto-discovers all outputs_* directories.",
    ),
):
    if output_dirs:
        dirs = output_dirs
    else:
        dirs = sorted(
            d for d in Path(".").iterdir()
            if d.is_dir() and d.name.startswith("outputs") and (d / "rewards").is_dir()
        )
        if not dirs:
            typer.echo("No outputs_* directories with a rewards/ subdirectory found.", err=True)
            raise typer.Exit(1)

    results: list[tuple[str, int, int]] = []

    for d in dirs:
        if not d.is_dir():
            typer.echo(f"Warning: '{d}' is not a directory, skipping", err=True)
            continue
        rewards_dir = d / "rewards"
        if not rewards_dir.is_dir():
            typer.echo(f"Warning: '{d}' has no 'rewards/' subdirectory, skipping", err=True)
            continue
        reward, rejected = _scan(rewards_dir)
        _print_table(f"Reward Rate — {d.name}", reward, rejected)
        results.append((d.name, reward, rejected))

    if not results:
        typer.echo("No valid output directories found.", err=True)
        raise typer.Exit(1)

    if len(results) > 1:
        table = Table(title="Collective Summary")
        table.add_column("Rewards", style="cyan")
        table.add_column("Accepted", justify="right")
        table.add_column("Rejected", justify="right")
        table.add_column("Total", justify="right")
        table.add_column("Accept Rate", justify="right")

        for name, reward, rejected in results:
            total = reward + rejected
            rate = (reward / total * 100) if total > 0 else 0.0
            table.add_row(name, str(reward), str(rejected), str(total), f"{rate:.2f}%")

        console.print()
        console.print(table)


if __name__ == "__main__":
    typer.run(main)
