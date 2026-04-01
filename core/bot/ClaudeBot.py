from pathlib import Path

import anthropic

from core.bot.BaseSC2Bot import BaseSC2Bot
from core.settings import get_settings
from core import console

_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompt.txt"
_SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")


class ClaudeBot(BaseSC2Bot):
    """
    SC2 bot that calls the Anthropic Messages API.
    Set ANTHROPIC_API_KEY and TACBENCH_MODEL_NAME in .env.
    """

    name: str = "ClaudeBot"

    def __init__(self):
        super().__init__()
        api_key = get_settings().anthropic_api_key
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set in .env or environment.")
        self._client_handler = anthropic.AsyncAnthropic(api_key=api_key)

    async def get_new_directive_async(self, current_battlefield_obs: str, step: int = 0) -> str:
        console.print_prompting(step)
        message = await self._client_handler.messages.create(
            model=self.MODEL_NAME,
            max_tokens=256,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": current_battlefield_obs}],
        )
        return message.content[0].text
