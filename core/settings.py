from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class TacBenchSettings(BaseSettings):
    """
    Central configuration loaded from .env and/or OS environment variables.
    All settings can be overridden by prefixing with TACBENCH_ in your .env file.
    For example: TACBENCH_K_STEPS=50
    """
    model_config = SettingsConfigDict(env_file=".env", env_prefix="TACBENCH_")

    map: str = "tacbench_01"
    player_race: str = "Terran"
    opponent_race: str = "Terran"
    opponent_difficulty: str = "Easy"
    realtime: bool = False
    k_steps: int = 30
    max_steps: int = 1000
    fallback_directive: str = "HOLD_POSITION"

    # Observation section toggles — set via TACBENCH_SHOW_* in .env
    show_your_units: bool = True
    show_your_structures: bool = True
    show_enemy_units: bool = True
    show_enemy_structures: bool = True
    show_supply: bool = True
    show_game_time: bool = True
    show_locations: bool = True
    show_tactical_overview: bool = True  # per-cluster matchups with range labels and local force ratios
    cluster_radius: float = 12.0         # tile radius used to group units into clusters

    # Historical context fed to the LLM alongside the current observation
    show_history: bool = False
    history_length: int = 3  # number of past LLM calls to include

    # Terrain map appended to each observation (static ASCII grid from game_info).
    # Warning: adds ~500-1500 tokens depending on map size and downsample factor.
    show_terrain: bool = False
    terrain_downsample: int = 4  # higher = fewer tokens, less detail

    # Ollama model name used by OllamaBot and FairlibBot.
    model_name: str = "qwen3:8b"


_settings: Optional[TacBenchSettings] = None


def get_settings() -> TacBenchSettings:
    """
    Return a cached settings instance.
    """
    global _settings
    if _settings is None:
        _settings = TacBenchSettings()
    return _settings
