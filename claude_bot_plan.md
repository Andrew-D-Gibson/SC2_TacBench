# Plan: Add ClaudeBot for Anthropic API inference

## Context
Add a `ClaudeBot` implementation that calls the Anthropic Messages API instead of a
local Ollama server, for better reasoning quality with reasonable latency.

**Latency:** With `WAIT_FOR_LLM=false` the game loop never blocks — it runs with the
cached directive and applies the response whenever it arrives. Claude Haiku typically
responds in 0.5–1.5 s; Sonnet in 1–3 s. Both fit within the 5 s timeout.

**API key note:** Claude Pro subscription and Anthropic API billing are separate.
An API key is required from platform.anthropic.com. Token costs here are negligible
(~300–500 tokens/call, every 15 game steps).

---

## Files to modify

| File | Change |
|------|--------|
| `core/ClaudeBot.py` | **New file** — Anthropic SDK bot |
| `core/main.py` | Select bot via `bot_type` setting |
| `core/settings.py` | Add `bot_type` field |
| `requirements.txt` | Add `anthropic>=0.40.0` |
| `.env` | Add `TACBENCH_BOT_TYPE=claude` and model name |
| `.env.example` | Document `ANTHROPIC_API_KEY` and Claude model options |

---

## 1. `core/ClaudeBot.py` (new file)

```python
from pathlib import Path
import anthropic
from BaseSC2Bot import BaseSC2Bot

_PROMPT_PATH = Path(__file__).parent.parent / "prompt.txt"
_SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")


class ClaudeBot(BaseSC2Bot):
    """SC2 bot that calls the Anthropic Messages API."""
    name: str = "ClaudeBot"

    def __init__(self):
        super().__init__()
        # Reads ANTHROPIC_API_KEY from environment automatically
        self._client = anthropic.AsyncAnthropic()

    async def get_new_directive_async(self, current_battlefield_obs: str) -> str:
        message = await self._client.messages.create(
            model=self.MODEL_NAME,
            max_tokens=256,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": current_battlefield_obs}],
        )
        return message.content[0].text
```

Key design choices:
- Uses `system=` param (proper Claude API pattern vs. concatenating into user message like OllamaBot)
- `max_tokens=256` — directive JSON is small; caps spend and latency
- No warmup needed (API is always ready, unlike a local Ollama server)
- `MODEL_NAME` comes from `TACBENCH_MODEL_NAME` in `.env`

---

## 2. `core/settings.py` — add `bot_type`

```python
# Which bot implementation to use: "ollama", "claude", or "fairlib"
bot_type: str = "ollama"
```

---

## 3. `core/main.py` — bot selection logic

Replace the hardcoded `bot = OllamaBot()` block with:

```python
from ClaudeBot import ClaudeBot

bot_type = settings.bot_type.lower()
if bot_type == "claude":
    bot = ClaudeBot()
elif bot_type == "fairlib":
    bot = FairlibBot()
else:
    bot = OllamaBot()
    asyncio.run(bot.warmup())
```

Warmup only runs for Ollama (local server needs to pre-load the model).

---

## 4. `requirements.txt`

Add:
```
anthropic>=0.40.0
```

---

## 5. `.env` additions

```
TACBENCH_BOT_TYPE=claude
TACBENCH_MODEL_NAME=claude-haiku-4-5-20251001
ANTHROPIC_API_KEY=sk-ant-...
```

---

## 6. `.env.example` additions

Under `# --- LLM ---`:
```
# TACBENCH_BOT_TYPE=ollama              # ollama | claude | fairlib
# ANTHROPIC_API_KEY=sk-ant-...          # required when BOT_TYPE=claude
# TACBENCH_MODEL_NAME=qwen3:8b                      # Ollama
# TACBENCH_MODEL_NAME=claude-haiku-4-5-20251001     # Claude (fast, ~1s)
# TACBENCH_MODEL_NAME=claude-sonnet-4-6             # Claude (smarter, ~2s)
```

---

## Recommended starting model

**`claude-haiku-4-5-20251001`** — fastest Claude model, excellent JSON instruction-following.
Upgrade to `claude-sonnet-4-6` for stronger tactical reasoning.

---

## Verification

1. Add `ANTHROPIC_API_KEY` to `.env` or OS environment
2. Set `TACBENCH_BOT_TYPE=claude` and `TACBENCH_MODEL_NAME=claude-haiku-4-5-20251001` in `.env`
3. Run `python core/main.py` — confirm `[TacBench] Game started with bot ClaudeBot`
4. Check final JSONL log: `llm_latency_ms` values and `"MODEL_NAME"` in the summary entry
