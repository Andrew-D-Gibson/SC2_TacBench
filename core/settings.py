from typing import Optional
from pydantic import Field
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
    show_tactical_overview: bool = True  # per-cluster matchups with range/strength/velocity info
    cluster_radius: float = 12.0         # tile radius used to group units into clusters
    cluster_track_interval: int = 5      # update cluster velocities every N game steps

    # Print the full LLM input to the console each call (useful for debugging prompts)
    show_llm_prompt: bool = False

    # Historical context fed to the LLM alongside the current observation
    show_history: bool = False
    history_length: int = 3  # number of past LLM calls to include

    # Terrain map appended to each observation (static ASCII grid from game_info).
    # Warning: adds ~500-1500 tokens depending on map size and downsample factor.
    show_terrain: bool = False
    terrain_downsample: int = 4  # higher = fewer tokens, less detail

    # Which bot implementation to use: "ollama", "claude", or "fairlib"
    bot_type: str = "ollama"

    # Model name passed to the selected bot.
    # Ollama example: "qwen3:8b"
    # Claude examples: "claude-haiku-4-5-20251001", "claude-sonnet-4-6"
    model_name: str = "qwen3:8b"

    # Anthropic API key — read from ANTHROPIC_API_KEY (no TACBENCH_ prefix).
    # pydantic-settings does not inject .env values into os.environ, so we
    # capture it here and pass it explicitly to AsyncAnthropic in ClaudeBot.
    anthropic_api_key: Optional[str] = Field(default=None, validation_alias="ANTHROPIC_API_KEY")


_settings: Optional[TacBenchSettings] = None


def get_settings() -> TacBenchSettings:
    """
    Return a cached settings instance.
    """
    global _settings
    if _settings is None:
        _settings = TacBenchSettings()
    return _settings
