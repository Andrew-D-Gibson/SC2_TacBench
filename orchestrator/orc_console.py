"""
orc_console.py — Rich-formatted console output for the SC2 orchestrator loop.

Mirrors the role of core/console.py (ANSI, for the game bot) but uses the
rich library for tables, panels, and rules suited to the higher-level loop.
"""

from datetime import datetime

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

_c = Console(highlight=False)


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _outcome_style(outcome: str) -> str:
    if outcome == "WIN":     return "bold green"
    if outcome == "LOSS":    return "bold red"
    if outcome == "TIMEOUT": return "bold yellow"
    return "white"


# ── Startup ────────────────────────────────────────────────────────────────────

def startup(maps: list, max_iterations: int, max_stagnant: int):
    _c.print()
    _c.print(Panel(
        f"[bold cyan]SC2 Prompt Optimization Orchestrator[/bold cyan]\n"
        f"[dim]Maps:[/dim] [white]{', '.join(maps)}[/white]   "
        f"[dim]Max iterations:[/dim] [white]{max_iterations}[/white]   "
        f"[dim]Max stagnant:[/dim] [white]{max_stagnant}[/white]",
        border_style="cyan",
        expand=False,
    ))
    _c.print()


# ── Iteration header ───────────────────────────────────────────────────────────

def iteration_header(iteration: int, max_iterations: int, stagnant: int, max_stagnant: int):
    stagnant_str = (
        f"  [yellow]stagnant {stagnant}/{max_stagnant}[/yellow]"
        if stagnant > 0 else ""
    )
    _c.print()
    _c.print(Rule(
        f"[bold white] ITERATION {iteration} / {max_iterations}{stagnant_str} [/bold white]",
        style="bright_blue",
    ))
    _c.print()


# ── Map run ────────────────────────────────────────────────────────────────────

def map_start(map_id: str):
    _c.print(f"[dim]{_ts()}[/dim]  [cyan]▶[/cyan]  Running map [bold]{map_id}[/bold]…")


def map_done(map_id: str, elapsed: float, exit_code: int):
    _c.print(
        f"[dim]{_ts()}[/dim]  [green]✓[/green]  Map [bold]{map_id}[/bold] "
        f"finished in [white]{elapsed:.0f}s[/white] "
        f"[dim](exit {exit_code})[/dim]"
    )


def map_error(map_id: str, msg: str):
    _c.print(f"[dim]{_ts()}[/dim]  [bold red]✗[/bold red]  Map [bold]{map_id}[/bold]: {msg}")


# ── Results table ──────────────────────────────────────────────────────────────

def results_table(results: dict, title: str = "Run Results"):
    table = Table(
        title=title, box=box.ROUNDED,
        border_style="dim", title_style="bold white",
        show_lines=False,
    )
    table.add_column("Map",     style="cyan", no_wrap=True)
    table.add_column("Outcome", justify="center")
    table.add_column("Steps",   justify="right", style="dim")
    table.add_column("Note",    style="dim")

    for map_id, r in results.items():
        outcome = r.get("outcome", "LOSS")
        table.add_row(
            map_id,
            Text(outcome, style=_outcome_style(outcome)),
            str(r.get("total_steps", 0)),
            r.get("error") or "",
        )
    _c.print(table)


# ── Meta-reasoner ──────────────────────────────────────────────────────────────

def meta_phase(phase: int, label: str, backend: str):
    _c.print()
    _c.print(Rule(
        f"[bold magenta] Phase {phase}: {label} [/bold magenta][dim]({backend})[/dim]",
        style="magenta",
    ))
    _c.print()


def llm_stream_header(label: str):
    """Printed just before streamed tokens arrive — no trailing newline."""
    _c.print(f"[dim magenta]  ┌─ {label}[/dim magenta]")
    _c.print(f"[dim magenta]  │[/dim magenta] ", end="")


def meta_decision_panel(action: str, reason: str):
    action_styles = {
        "edit_prompt":       "bold green",
        "edit_code":         "bold yellow",
        "stop_missing_info": "bold red",
        "noop":              "dim white",
    }
    style = action_styles.get(action, "white")
    _c.print()
    _c.print(Panel(
        f"[bold]Action:[/bold]  [{style}]{action}[/{style}]\n"
        f"[bold]Reason:[/bold]  [dim]{reason}[/dim]",
        title="[bold white]◆ Meta Decision[/bold white]",
        border_style="magenta",
        expand=False,
    ))


def meta_error(msg: str):
    _c.print(f"  [bold red]✗[/bold red]  [dim]{msg}[/dim]")


# ── File editor ────────────────────────────────────────────────────────────────

def edit_start(rel_path: str):
    _c.print(f"  [cyan]✎[/cyan]  Editing [bold]{rel_path}[/bold]…")


def edit_ok(rel_path: str):
    _c.print(f"  [green]✓[/green]  Written: [bold]{rel_path}[/bold]")


def edit_fail(rel_path: str, reason: str):
    _c.print(f"  [bold red]✗[/bold red]  Failed [bold]{rel_path}[/bold]: [dim]{reason}[/dim]")


def edit_blocked(rel_path: str):
    _c.print(f"  [bold red]⊘[/bold red]  Blocked: [bold]{rel_path}[/bold] [dim]not in whitelist[/dim]")


# ── Improvement / revert ───────────────────────────────────────────────────────

def kept(delta: dict):
    parts = []
    for m, v in delta.items():
        parts.append(f"{m}: [bold green]+{v}[/bold green]" if v > 0 else f"{m}: [dim]{v}[/dim]")
    _c.print()
    _c.print(Panel(
        f"[bold green]Changes committed.[/bold green]  Δ steps — {',  '.join(parts)}",
        title="[bold green]✓ IMPROVEMENT[/bold green]",
        border_style="green",
        expand=False,
    ))


def reverted(delta: dict, reason: str):
    parts = [f"{m}: [dim]{v}[/dim]" for m, v in delta.items()]
    _c.print()
    _c.print(Panel(
        f"[dim]Reason: {reason}[/dim]" +
        (f"\n Δ steps — {',  '.join(parts)}" if any(v != 0 for v in delta.values()) else ""),
        title="[bold red]↩ REVERTED[/bold red]",
        border_style="red",
        expand=False,
    ))


# ── Stop messages ──────────────────────────────────────────────────────────────

def stop(reason: str):
    msgs = {
        "all_maps_won":   ("[bold green]★  All maps won — optimization complete![/bold green]", "green"),
        "stagnant":       ("[bold yellow]⏹  Stagnation limit reached — no improvement.[/bold yellow]", "yellow"),
        "max_iterations": ("[bold white]⏹  Max iterations reached.[/bold white]", "white"),
        "missing_info":   ("[bold red]⏹  Stopped: missing info (see reason above).[/bold red]", "red"),
    }
    text, border = msgs.get(reason, (f"[white]Stopped: {reason}[/white]", "white"))
    _c.print()
    _c.print(Panel(text, border_style=border, expand=False))
    _c.print()


# ── Final summary ──────────────────────────────────────────────────────────────

def run_history_table(run_history: dict):
    table = Table(
        title="Final Run History", box=box.ROUNDED,
        border_style="cyan", title_style="bold white",
    )
    table.add_column("Map",         style="cyan", no_wrap=True)
    table.add_column("Best Steps",  justify="right", style="white")
    table.add_column("Best Result", justify="center")

    for map_id, h in run_history.items():
        result = h.get("best_result", "LOSS")
        table.add_row(
            map_id,
            str(h.get("best_steps", 0)),
            Text(result, style=_outcome_style(result)),
        )
    _c.print()
    _c.print(table)
    _c.print()


# ── Generic status / git ───────────────────────────────────────────────────────

def status(msg: str):
    _c.print(f"[dim]{_ts()}[/dim]  {msg}")


def git_msg(msg: str):
    _c.print(f"[dim]{_ts()}[/dim]  [dim cyan]git ·[/dim cyan] [dim]{msg}[/dim]")
