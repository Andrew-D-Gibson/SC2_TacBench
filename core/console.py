"""
console.py — Pretty-printed, ANSI-colored console output for TacBench.

Supported on Windows 10+ (virtual terminal processing) and all modern terminals.
Call console.init() once at startup to enable ANSI on Windows.
"""

import sys

# ── ANSI codes ─────────────────────────────────────────────────────────────────

RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"

RED     = "\033[31m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
BLUE    = "\033[34m"
MAGENTA = "\033[35m"
CYAN    = "\033[36m"

BRIGHT_RED    = "\033[91m"
BRIGHT_GREEN  = "\033[92m"
BRIGHT_YELLOW = "\033[93m"
BRIGHT_BLUE   = "\033[94m"
BRIGHT_CYAN   = "\033[96m"
BRIGHT_WHITE  = "\033[97m"

# ── Init ───────────────────────────────────────────────────────────────────────

def init():
    """Enable ANSI virtual terminal processing on Windows 10+."""
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            # ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004, combined with
            # ENABLE_PROCESSED_OUTPUT (0x0001) and ENABLE_WRAP_AT_EOL_OUTPUT (0x0002)
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass  # Best-effort; modern terminals (VS Code, Windows Terminal) work without it.

# ── Internal helpers ───────────────────────────────────────────────────────────

_W = 80  # default console width for dividers


def _rule(char: str = "─", width: int = _W) -> str:
    return char * width


def _trunc(s: str, max_len: int) -> str:
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def _directive_color(name: str, fallback_used: bool) -> str:
    """Pick a color for a directive name."""
    if fallback_used:
        return RED
    if name in ("ATTACK", "FOCUS_FIRE"):
        return BRIGHT_RED
    if name == "MOVE":
        return BRIGHT_YELLOW
    if name in ("HOLD_POSITION", "SPREAD"):
        return BRIGHT_CYAN
    return BRIGHT_WHITE


# ── Public API ─────────────────────────────────────────────────────────────────

def print_startup_banner(bot_name: str, model_name: str):
    """Large banner printed once at startup."""
    w = _W
    title = f"  StarCraft II TacBench  ──  {bot_name}  ──  {model_name}  "
    print()
    print(f"{BOLD}{CYAN}{'═' * w}{RESET}")
    print(f"{BOLD}{CYAN}{title.center(w)}{RESET}")
    print(f"{BOLD}{CYAN}{'═' * w}{RESET}")
    print()


def print_warmup(model_name: str, done: bool = False, error_msg: str = None):
    if error_msg:
        print(f"{RED}  ✗  Warmup failed ({model_name}): {error_msg}{RESET}\n")
    elif done:
        print(f"{GREEN}  ✓  Warmup complete  ({model_name}){RESET}\n")
    else:
        print(f"{CYAN}  ·  Warming up {BOLD}{model_name}{RESET}{CYAN}…{RESET}", flush=True)


def print_game_start(map_name: str, race):
    """Compact game-start block."""
    w = _W
    print(f"{BOLD}{GREEN}{_rule('─', w)}{RESET}")
    print(f"{BOLD}{GREEN}  GAME STARTED{RESET}")
    print(f"{GREEN}  Map:   {BOLD}{map_name}{RESET}")
    print(f"{GREEN}  Race:  {BOLD}{race}{RESET}")
    print(f"{BOLD}{GREEN}{_rule('─', w)}{RESET}\n")


def print_llm_prompt(step: int, model_name: str, prompt: str):
    """
    Prints the full prompt being sent to the LLM.
    Shows a decorated header, every prompt line, then a footer rule.
    """
    char_count = len(prompt)
    tag = f" LLM CALL @ Step {step}  ──  {model_name}  ──  {char_count:,} chars "
    pad = max(0, _W - len(tag))
    print(f"\n{BOLD}{BLUE}{'━' * 4}{tag}{'━' * pad}{RESET}")
    for line in prompt.splitlines():
        print(f"{DIM}  {line}{RESET}")
    print(f"{BLUE}{'━' * _W}{RESET}")


def print_prompting(step: int):
    print(f"\n{BOLD}{YELLOW}▶ Prompting... (Step {step}):{RESET}  ", end="", flush=True)
    

def print_streaming_start(step: int):
    """Prefix printed just before streaming tokens arrive — no newline."""
    print(f"\n{BOLD}{YELLOW}▶ Response (Step {step}):{RESET}  ", end="", flush=True)


def print_streaming_end():
    """Newline after the last streamed token."""
    print()


def print_directive(step: int, directive, friendly: int, enemy: int, latency_ms: int):
    """
    Colored summary for a resolved directive, printed after each LLM call completes.
    """
    name = directive.name
    col  = _directive_color(name, directive.fallback_used)

    header_tag = f" DIRECTIVE @ Step {step}  ──  {latency_ms:,} ms "
    pad = max(0, _W - len(header_tag))
    print(f"\n{BOLD}{col}{'▶' * 2}{header_tag}{'─' * pad}{RESET}")

    # Main directive name
    prefix = f"{BOLD}{col}  ► {RESET}"
    print(f"{prefix}{BOLD}{col}{name}{RESET}", end="")
    if directive.target_x is not None and directive.target_y is not None:
        print(f"  {DIM}→ ({directive.target_x:.1f}, {directive.target_y:.1f}){RESET}", end="")
    print()

    # Reasoning
    if directive.reasoning:
        r = _trunc(directive.reasoning, _W - 5)
        print(f"  {DIM}\"{r}\"{RESET}")

    # Error / fallback notice
    if directive.error:
        print(f"  {BOLD}{RED}⚠  {directive.error}{RESET}")
    elif directive.fallback_used:
        print(f"  {YELLOW}⚠  fallback used{RESET}")

    # Battle stats
    friendly_str = f"{BRIGHT_CYAN}Friendly: {friendly}{RESET}"
    enemy_str    = f"{BRIGHT_RED}Enemy visible: {enemy}{RESET}"
    print(f"  {friendly_str}    {enemy_str}")
    print(f"{col}{'─' * _W}{RESET}\n")


def print_game_over(outcome: str, total_steps: int, llm_calls: int):
    if outcome == "WIN":
        col, icon = BRIGHT_GREEN, "★"
    elif outcome == "LOSS":
        col, icon = BRIGHT_RED, "✗"
    else:
        col, icon = YELLOW, "─"

    w = _W
    title = f"  {icon}  GAME OVER: {outcome}  {icon}  "
    print(f"\n{BOLD}{col}{'═' * w}{RESET}")
    print(f"{BOLD}{col}{title.center(w)}{RESET}")
    print(f"{col}{'─' * w}{RESET}")
    print(f"{col}  Total Steps : {total_steps:,}{RESET}")
    print(f"{col}  LLM Calls   : {llm_calls:,}{RESET}")
    print(f"{BOLD}{col}{'═' * w}{RESET}\n")


def print_log_saved(path: str):
    print(f"{DIM}  Log → {path}{RESET}\n")


def warn(msg: str):
    print(f"{YELLOW}  ⚠  {msg}{RESET}")


def error(msg: str):
    print(f"{BOLD}{RED}  ✗  {msg}{RESET}")
