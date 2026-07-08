import typer
from pathlib import Path
from rich.table import Table
from rich.console import Console

console = Console()


def _scan_dir(rewards_dir: Path) -> tuple[int, int]:
    rejected = sum(1 for f in rewards_dir.iterdir() if f.is_file() and f.name.startswith("rejected"))
    reward = sum(1 for f in rewards_dir.iterdir() if f.is_file() and f.name.startswith("reward"))
    return reward, rejected


def _print_single(output_dir: Path):
    rewards_dir = output_dir / "rewards"
    if not rewards_dir.is_dir():
        typer.echo(f"Error: '{output_dir}' does not contain a 'rewards/' subdirectory", err=True)
        raise typer.Exit(1)

    reward, rejected = _scan_dir(rewards_dir)
    total = reward + rejected
    rate = (reward / total * 100) if total > 0 else 0.0

    table = Table(title=f"Reward Rate — {rewards_dir}")
    table.add_column("Category", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Percentage", justify="right")
    table.add_row("Reward", str(reward), f"{rate:.2f}%")
    table.add_row("Rejected", str(rejected), f"{100 - rate:.2f}%")
    table.add_row("Total", str(total), "100.00%")
    console.print(table)


def _print_all():
    dirs = sorted(d for d in Path(".").iterdir() if d.is_dir() and d.name.startswith("outputs"))
    rows = []
    for d in dirs:
        if not d.is_dir():
            continue
        rewards_dir = d / "rewards"
        if not rewards_dir.is_dir():
            continue
        reward, rejected = _scan_dir(rewards_dir)
        total = reward + rejected
        rate = (reward / total * 100) if total > 0 else 0.0
        rows.append((d.name, reward, rejected, total, rate))

    if not rows:
        typer.echo("No outputs_* directories with a rewards/ subdirectory found.", err=True)
        raise typer.Exit(1)

    # Per-dir tables
    for name, reward, rejected, total, rate in rows:
        t = Table(title=name)
        t.add_column("Category", style="cyan")
        t.add_column("Count", justify="right")
        t.add_column("Percentage", justify="right")
        t.add_row("Reward", str(reward), f"{rate:.2f}%")
        t.add_row("Rejected", str(rejected), f"{100 - rate:.2f}%")
        t.add_row("Total", str(total), "100.00%")
        console.print(t)
        console.print()

    # Combined table
    table = Table(title="Combined — All Output Directories")
    table.add_column("Output Directory", style="cyan")
    table.add_column("Rewards", justify="right")
    table.add_column("Rejected", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Accept Rate", justify="right")

    grand_r = grand_j = 0
    for name, reward, rejected, total, rate in rows:
        table.add_row(name, str(reward), str(rejected), str(total), f"{rate:.1f}%")
        grand_r += reward
        grand_j += rejected

    grand_t = grand_r + grand_j
    table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{grand_r}[/bold]",
        f"[bold]{grand_j}[/bold]",
        f"[bold]{grand_t}[/bold]",
        f"[bold]{(grand_r / grand_t * 100):.1f}%[/bold]" if grand_t > 0 else "[bold]N/A[/bold]",
    )
    console.print(table)


def main(
    output_dir: Path = typer.Argument(
        None,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Path to the output directory (must contain a rewards/ subdirectory). "
        "If omitted, scans all outputs_* directories.",
    ),
):
    if output_dir is None:
        _print_all()
    else:
        _print_single(output_dir)


if __name__ == "__main__":
    typer.run(main)
