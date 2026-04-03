#!/usr/bin/env python3
"""
replay_observer.py — Watch a TacBench SC2 replay with LLM decision overlay.

Loads a paired JSONL episode log and, at every step where the original bot
made an LLM call, prints the battlefield observation and directives to the
console — synchronized with the replay playing in the StarCraft II window.

Usage
-----
Interactive (pick from lists):
    python replay/replay_observer.py

Explicit replay + auto-matched log:
    python replay/replay_observer.py "C:/path/to/replay.SC2Replay"

Fully explicit:
    python replay/replay_observer.py "C:/path/to/replay.SC2Replay" --log "logs/my_log.jsonl"

Options:
    --log <path>          Path to the JSONL episode log to overlay.
    --speed SPEED         Playback speed (default: faster = 22.4 gl/s, SC2 competitive speed).
    --show-battlefield    Also print the stored battlefield text at each LLM step.
    --observed-id <int>   Which player ID to observe in the replay (default: 0).

Step-count matching
-------------------
burnysc2 hardcodes game_step=1 for replays, which makes each on_step() call
correspond to one raw game tick.  The observer matches log entries by
game_loop value, computed as:

    game_loop = (bot_step - 1) * original_game_step

The correct original_game_step is read automatically from the log's summary
entry (field config.game_step).  For logs that predate that field, it falls
back to --original-game-step (default: 1).
"""

from __future__ import annotations

import sys
import json
import argparse
from io import BytesIO
from pathlib import Path

# Make project root importable regardless of working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import time

import mpyq
from sc2.observer_ai import ObserverAI
from sc2.data import Result

from core import console

# ── Constants ─────────────────────────────────────────────────────────────────

# game_step for the replay observer.  1 gives the smoothest visual playback
# because the SC2 engine advances one tick at a time.
_DEFAULT_GAME_STEP = 1

# game_step the original bot used during recording.
# Used to convert log step numbers → expected game_loop values for matching.
# burnysc2's Client.game_step default is 4; override with --original-game-step
# if you know the episode used a different value.
_DEFAULT_ORIGINAL_GAME_STEP = 4

_REPLAY_DIR = Path(r"C:\Users\adgib\OneDrive\Documents\StarCraft II\Replays\Multiplayer")
_LOG_DIR    = Path("logs")

# SC2 game speeds in game-loops per real second.
# Derived from: Normal = 16 gl/s, with speed multipliers applied.
#   Slower = 0.50×, Slow = 0.75×, Normal = 1.0×, Fast = 1.2×, Faster = 1.4×
# "Faster" (22.4) is the competitive/ladder standard and what burnysc2 assumes.
GAME_SPEEDS: dict[str, float] = {
    "slower": 8.0,
    "slow":   12.0,
    "normal": 16.0,
    "fast":   19.2,
    "faster": 22.4,
}
_SPEED_CHOICES = ["max"] + list(GAME_SPEEDS.keys())


# ── Replay metadata helpers ────────────────────────────────────────────────────

def _parse_replay_metadata(replay_path: Path) -> dict:
    """Extract metadata dict from a .SC2Replay file using mpyq."""
    with open(replay_path, "rb") as f:
        data = f.read()
    archive = mpyq.MPQArchive(BytesIO(data)).extract()
    return json.loads(archive[b"replay.gamemetadata.json"].decode("utf-8"))


def _replay_map_name(replay_path: Path) -> str:
    """Return the map name stored in the replay's metadata."""
    try:
        meta = _parse_replay_metadata(replay_path)
        return meta.get("Title", replay_path.stem)
    except Exception:
        return replay_path.stem


# ── Log helpers ────────────────────────────────────────────────────────────────

def _find_matching_log(map_name: str, log_dir: Path) -> Path | None:
    """
    Return the most recently modified log file whose filename contains the
    normalised map name (lowercase, spaces → underscores).
    """
    if not log_dir.exists():
        return None
    norm = map_name.lower().replace(" ", "_")
    candidates = sorted(
        [p for p in log_dir.glob("*.jsonl") if norm in p.name.lower()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _load_log(log_path: Path) -> tuple[dict, dict[int, dict]]:
    """
    Parse a TacBench episode log.

    Each entry is written with json.dumps(..., indent=4), so objects span
    multiple lines.  We use JSONDecoder.raw_decode to extract them one by
    one from the raw file content rather than reading line-by-line.

    Returns
    -------
    summary          : dict — the first "summary" entry (outcome, config, etc.)
    calls_by_step    : dict[int, dict] — step → llm_call entry
    """
    decoder = json.JSONDecoder()
    summary: dict = {}
    calls_by_step: dict[int, dict] = {}

    text = log_path.read_text(encoding="utf-8")
    idx  = 0
    while idx < len(text):
        # Skip whitespace / newlines between objects.
        while idx < len(text) and text[idx] in " \t\r\n":
            idx += 1
        if idx >= len(text):
            break
        entry, end = decoder.raw_decode(text, idx)
        idx = end
        t = entry.get("type")
        if t == "summary":
            summary = entry
        elif t == "llm_call":
            calls_by_step[entry["step"]] = entry

    return summary, calls_by_step


# ── Interactive selection helpers ──────────────────────────────────────────────

def _pick_from_list(items: list[Path], label: str) -> Path | None:
    """Print a numbered list and prompt the user to pick one."""
    if not items:
        print(f"{console.YELLOW}  No {label} found.{console.RESET}")
        return None
    print(f"\n{console.BOLD}{console.CYAN}  Recent {label}:{console.RESET}")
    for i, p in enumerate(items, 1):
        print(f"  {console.BRIGHT_WHITE}{i:2}.{console.RESET} {p.name}")
    while True:
        raw = input(f"\n  Select {label} [1-{len(items)}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(items):
            return items[int(raw) - 1]
        print(f"  {console.YELLOW}Invalid choice — enter a number between 1 and {len(items)}.{console.RESET}")


def _list_recent(directory: Path, glob: str, n: int = 10) -> list[Path]:
    """Return the N most recently modified files matching a glob in directory."""
    if not directory.exists():
        return []
    return sorted(directory.glob(glob), key=lambda p: p.stat().st_mtime, reverse=True)[:n]


# ── Custom replay runner ──────────────────────────────────────────────────────
#
# SC2 replays have no built-in realtime pacing in burnysc2 — the engine always
# processes steps as fast as Python requests them.  To replay at a target speed
# we call client.step() every iteration (to advance the visual) and sleep for
# whatever wall-clock time remains in the target frame interval.
#
# Jitter is minimised by two techniques:
#   1. Deadline-based sleep: track the absolute target time for the next frame
#      so late iterations shorten the next sleep rather than accumulating drift.
#   2. Windows timer resolution: timeBeginPeriod(1) gives ~1 ms sleep granularity
#      instead of the default ~15 ms, called once in main() before the loop.

async def _play_replay_tacbench(
    client,
    ai: "TacBenchReplayObserver",
    ticks_per_second: float | None,  # None = max speed (no sleep)
):
    """
    Replay loop that matches burnysc2's _play_game_ai realtime handling:
      - realtime=True : skip client.step(); SC2 advances on its own clock and
                        client.observation(game_loop=N) blocks until tick N.
      - realtime=False: call client.step() to advance as fast as possible.
    """
    from sc2.game_state import GameState
    from sc2.protocol import ProtocolError
    from s2clientprotocol import sc2api_pb2 as sc_pb

    ai._initialize_variables()

    game_data   = await client.get_game_data()
    game_info   = await client.get_game_info()
    ping_resp   = await client.ping()

    # Set game_step before the loop so observation timing is consistent.
    client.game_step = ai._game_step

    ai._prepare_start(
        client, 0, game_info, game_data,
        realtime=False, base_build=ping_resp.ping.base_build,
    )

    state = await client.observation()
    if client._game_result:
        result = next(iter(client._game_result.values()), Result.Defeat)
        await ai.on_end(result)
        return result

    gs              = GameState(state.observation)
    proto_game_info = await client._execute(game_info=sc_pb.RequestGameInfo())
    ai._prepare_step(gs, proto_game_info)
    ai._prepare_first_step()

    try:
        await ai.on_start()
    except Exception:
        await ai.on_end(Result.Defeat)
        return Result.Defeat

    # Deadline-based pacing: track the absolute wall-clock time the next frame
    # should start.  If an iteration runs long, the next sleep is shorter to
    # compensate, preventing cumulative drift.
    step_interval = (client.game_step / ticks_per_second) if ticks_per_second else None
    next_deadline = time.perf_counter()

    iteration = 0
    while True:
        if iteration != 0:
            state = await client.observation()

            if client._game_result:
                result = next(iter(client._game_result.values()), Result.Defeat)
                await ai.on_end(result)
                return result

            gs              = GameState(state.observation)
            proto_game_info = await client._execute(game_info=sc_pb.RequestGameInfo())
            ai._prepare_step(gs, proto_game_info)

        try:
            await ai.issue_events()
            await ai.on_step(iteration)
            await ai._after_step()
        except Exception as exc:
            if isinstance(exc, ProtocolError) and exc.is_game_over_error:
                await ai.on_end(Result.Victory)
                return Result.Victory
            import traceback
            print(f"\n[replay] Exception in step {iteration}: {exc}")
            traceback.print_exc()
            await ai.on_end(Result.Defeat)
            return Result.Defeat

        if not client.in_game:
            await ai.on_end(Result.Victory)
            return Result.Victory

        await client.step()

        if step_interval is not None:
            next_deadline += step_interval
            remaining = next_deadline - time.perf_counter()
            if remaining > 0:
                await asyncio.sleep(remaining)
            else:
                # We're already behind — reset the deadline to now so we don't
                # try to catch up by skipping sleep on all subsequent frames.
                next_deadline = time.perf_counter()

        iteration += 1


async def _host_replay_tacbench(
    replay_path: str,
    ai: "TacBenchReplayObserver",
    ticks_per_second: float | None,
    base_build: str,
    data_version: str,
    observed_id: int,
):
    from sc2.sc2process import SC2Process
    from sc2.client import Client

    async with SC2Process(fullscreen=False, base_build=base_build, data_hash=data_version) as server:
        # Pass realtime=False to SC2 — we control pacing entirely from Python.
        await server.start_replay(replay_path, False, observed_id)
        client = Client(server._ws)
        return await _play_replay_tacbench(client, ai, ticks_per_second)


def run_replay_tacbench(
    ai: "TacBenchReplayObserver",
    replay_path: Path | str,
    ticks_per_second: float | None = None,
    observed_id: int = 0,
):
    """
    Run an SC2 replay with optional speed control.

    ticks_per_second : float | None
        Target game-loop rate.  Use values from GAME_SPEEDS (e.g. 22.4 for
        "faster") or None to run at maximum speed.
    """
    from sc2.main import get_replay_version

    replay_path = str(replay_path)
    assert Path(replay_path).is_file(),     f"Replay not found: {replay_path}"
    assert Path(replay_path).is_absolute(), f"Replay path must be absolute: {replay_path}"

    base_build, data_version = get_replay_version(replay_path)
    return asyncio.get_event_loop().run_until_complete(
        _host_replay_tacbench(replay_path, ai, ticks_per_second, base_build, data_version, observed_id)
    )


# ── Observer ──────────────────────────────────────────────────────────────────

class TacBenchReplayObserver(ObserverAI):
    """
    Observes an SC2 replay and overlays TacBench LLM data from a paired log.

    At each step that corresponds to an LLM call in the original episode, the
    battlefield observation, directives, and reasoning are printed to the
    console — giving a side-by-side view of what the LLM saw and decided.
    """

    def __init__(
        self,
        log_path: Path | None,
        game_step: int          = _DEFAULT_GAME_STEP,
        original_game_step: int = _DEFAULT_ORIGINAL_GAME_STEP,
        show_battlefield: bool  = False,
    ):
        super().__init__()
        self._log_path          = log_path
        self._game_step         = game_step
        self._original_game_step = original_game_step
        self._show_battlefield  = show_battlefield
        self._log_summary: dict = {}

        # Key log entries by the game_loop at which they were originally produced.
        # game_loop at bot step S = (S - 1) * original_game_step
        # (bot step_count starts at 1; game_loop=0 before the first client.step()).
        self._log_by_game_loop: dict[int, dict] = {}

        if log_path and log_path.exists():
            summary, by_step = _load_log(log_path)
            self._log_summary = summary
            # Prefer game_step embedded in the log over the CLI default.
            logged_gs = summary.get("config", {}).get("game_step")
            if logged_gs is not None:
                self._original_game_step = logged_gs
            self._log_by_game_loop = {
                (step - 1) * self._original_game_step: entry
                for step, entry in by_step.items()
            }

    async def on_start(self) -> None:
        console.print_replay_banner(
            map_name=self.game_info.map_name,
            log_name=self._log_path.name if self._log_path else "(no log paired)",
            summary=self._log_summary,
            total_log_steps=len(self._log_by_game_loop),
        )

    async def on_step(self, iteration: int) -> None:
        # Match log entries by game_loop, not iteration count.
        # game_loop is independent of the observer's game_step setting.
        entry = self._log_by_game_loop.get(self.state.game_loop)
        if entry:
            console.print_replay_llm_step(
                step=entry["step"],
                game_time=self.time_formatted,
                entry=entry,
                show_battlefield=self._show_battlefield,
            )

    async def on_end(self, game_result: Result) -> None:
        outcome     = self._log_summary.get("outcome", "?")
        total_steps = self._log_summary.get("total_steps", "?")
        total_calls = self._log_summary.get("total_llm_calls", len(self._log_by_game_loop))
        console.print_replay_end(outcome, total_steps, total_calls)


# ── Entry point ───────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Watch a TacBench SC2 replay with LLM decision overlay.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "replay",
        nargs="?",
        help="Path to the .SC2Replay file (absolute or relative). "
             "Omit to pick interactively.",
    )
    p.add_argument(
        "--log",
        metavar="PATH",
        help="Path to the paired JSONL episode log. "
             "Omit to auto-match from logs/ by map name.",
    )
    p.add_argument(
        "--speed",
        choices=_SPEED_CHOICES,
        default="faster",
        metavar="SPEED",
        help=(
            "Playback speed. "
            f"Choices: {', '.join(_SPEED_CHOICES)}. "
            "'max' runs as fast as possible (default). "
            "'faster' = 22.4 gl/s (SC2 competitive speed). "
            "'normal' = 16 gl/s. 'slow' = 12 gl/s. 'slower' = 8 gl/s."
        ),
    )
    p.add_argument(
        "--show-battlefield",
        action="store_true",
        help="Also print the stored battlefield text at each LLM step.",
    )
    p.add_argument(
        "--observed-id",
        type=int,
        default=0,
        metavar="N",
        help="Player slot to observe in the replay (default: 0).",
    )
    p.add_argument(
        "--game-step",
        type=int,
        default=_DEFAULT_GAME_STEP,
        metavar="N",
        help=f"Step size for the replay observer (default: {_DEFAULT_GAME_STEP}). "
             "1 gives the smoothest visual playback.",
    )
    p.add_argument(
        "--original-game-step",
        type=int,
        default=_DEFAULT_ORIGINAL_GAME_STEP,
        metavar="N",
        help=f"game_step the bot used when the episode was recorded (default: {_DEFAULT_ORIGINAL_GAME_STEP}). "
             "Used to align log entries with the correct game_loop in the replay. "
             "If the log contains a 'game_step' field in its summary, that value takes precedence.",
    )
    return p


def _set_timer_resolution():
    """
    On Windows, improve sleep granularity from ~15 ms to ~1 ms.
    This significantly reduces frame-time jitter when rate-limiting the replay.
    """
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.winmm.timeBeginPeriod(1)
        except Exception:
            pass


def main():
    console.init()
    _set_timer_resolution()
    parser = _build_parser()
    args   = parser.parse_args()

    # ── Resolve replay path ───────────────────────────────────────────────────
    if args.replay:
        replay_path = Path(args.replay)
        if not replay_path.is_absolute():
            # Try relative to cwd first, then to the default replay directory.
            if not replay_path.exists() and (_REPLAY_DIR / replay_path).exists():
                replay_path = _REPLAY_DIR / replay_path
            replay_path = replay_path.resolve()
    else:
        # Interactive: list recent replays from the default directory.
        recent_replays = _list_recent(_REPLAY_DIR, "*.SC2Replay")
        if not recent_replays:
            print(f"{console.RED}  No replays found in {_REPLAY_DIR}{console.RESET}")
            sys.exit(1)
        replay_path = _pick_from_list(recent_replays, "replays")
        if replay_path is None:
            sys.exit(1)
        replay_path = replay_path.resolve()

    if not replay_path.exists():
        print(f"{console.RED}  Replay not found: {replay_path}{console.RESET}")
        sys.exit(1)

    # ── Resolve log path ──────────────────────────────────────────────────────
    if args.log:
        log_path = Path(args.log).resolve()
        if not log_path.exists():
            print(f"{console.YELLOW}  Warning: log file not found: {log_path}{console.RESET}")
            log_path = None
    else:
        map_name  = _replay_map_name(replay_path)
        log_dir   = Path.cwd() / _LOG_DIR
        log_path  = _find_matching_log(map_name, log_dir)
        if log_path:
            print(f"\n{console.DIM}  Auto-matched log: {log_path.name}{console.RESET}")
        else:
            # Offer the user a chance to pick from recent logs.
            print(f"\n{console.YELLOW}  No log auto-matched for map '{map_name}'.{console.RESET}")
            recent_logs = _list_recent(Path.cwd() / _LOG_DIR, "*.jsonl")
            if recent_logs:
                log_path = _pick_from_list(recent_logs, "logs (or press Enter to skip)")
            # log_path may still be None — replay will run without overlay.

    # ── Run ───────────────────────────────────────────────────────────────────
    observer = TacBenchReplayObserver(
        log_path=log_path,
        game_step=args.game_step,
        original_game_step=args.original_game_step,
        show_battlefield=args.show_battlefield,
    )

    ticks_per_second = GAME_SPEEDS.get(args.speed)  # None when speed == "max"

    run_replay_tacbench(
        observer,
        replay_path=replay_path,
        ticks_per_second=ticks_per_second,
        observed_id=args.observed_id,
    )


if __name__ == "__main__":
    main()
