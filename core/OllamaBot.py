import asyncio
import json
import requests
from pathlib import Path

from BaseSC2Bot import BaseSC2Bot

OLLAMA_URL = "http://localhost:11434/api/chat"
STREAM = False  # set to False to wait for the full response without token-by-token output

# Load the shared prompt template once at import time.
# The prompt file lives alongside this script in core/.
_PROMPT_PATH = Path(__file__).parent.parent / "prompt.txt"
_SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")


def _stream_ollama(payload: dict) -> str:
    """
    Send a streaming request to Ollama, printing each token as it arrives.
    Returns the fully assembled response string when the stream completes.
    """
    full_text = ""
    with requests.post(OLLAMA_URL, json=payload, stream=True) as response:
        response.raise_for_status()
        print("[OllamaBot] ", end="", flush=True)
        for line in response.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            token = chunk.get("message", {}).get("content", "")
            print(token, end="", flush=True)
            full_text += token
            if chunk.get("done"):
                break
    print()  # newline after the stream finishes
    return full_text


class OllamaBot(BaseSC2Bot):
    """
    SC2 bot that calls the Ollama /api/chat endpoint directly via requests.
    Streams tokens to the terminal in real time, then returns the complete
    response for directive normalization. Runs a warmup call in on_start so
    Ollama is loaded and ready before the first timed game step.

    The system prompt is loaded from core/prompt.txt — edit that file to
    change the directive instructions without touching this class.
    """

    name: str = "OllamaBot"


    async def warmup(self):
        print("[OllamaBot] Warming up Ollama model...")
        payload = {
            "model": self.MODEL_NAME,
            "messages": [{"role": "user", "content": "ping"}],
            "stream": False,
        }
        try:
            await asyncio.to_thread(requests.post, OLLAMA_URL, json=payload)
            print("[OllamaBot] Warmup complete.\n")
        except Exception as exc:
            print(f"[OllamaBot] Warmup failed (continuing anyway): {exc}\n")


    async def get_new_directive_async(self, current_battlefield_obs: str) -> str:
        full_prompt = _SYSTEM_PROMPT + current_battlefield_obs
        payload = {
            "model": self.MODEL_NAME,
            "messages": [{"role": "user", "content": full_prompt}],
            "stream": STREAM,
            "think": False,
        }
        if STREAM:
            return await asyncio.to_thread(_stream_ollama, payload)
        response = await asyncio.to_thread(requests.post, OLLAMA_URL, json=payload)
        response.raise_for_status()
        return response.json()["message"]["content"]
