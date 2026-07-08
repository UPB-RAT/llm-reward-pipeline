import typer
from pathlib import Path
from rich.table import Table
from rich.console import Console

console = Console()


def main(
    output_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Path to the output directory (must contain a rewards/ subdirectory)",
    ),
):
    rewards_dir = output_dir / "rewards"
    if not rewards_dir.is_dir():
        typer.echo(f"Error: '{output_dir}' does not contain a 'rewards/' subdirectory", err=True)
        raise typer.Exit(1)

    rejected = sum(1 for f in rewards_dir.iterdir() if f.is_file() and f.name.startswith("rejected"))
    reward = sum(1 for f in rewards_dir.iterdir() if f.is_file() and f.name.startswith("reward"))
    total = rejected + reward
    reward_pct = (reward / total * 100) if total > 0 else 0.0
    rejected_pct = (rejected / total * 100) if total > 0 else 0.0

    table = Table(title=f"Reward Rate — {rewards_dir}")
    table.add_column("Category", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Percentage", justify="right")

    table.add_row("Reward", str(reward), f"{reward_pct:.2f}%")
    table.add_row("Rejected", str(rejected), f"{rejected_pct:.2f}%")
    table.add_row("Total", str(total), "100.00%")

    console.print(table)


if __name__ == "__main__":
    typer.run(main)
