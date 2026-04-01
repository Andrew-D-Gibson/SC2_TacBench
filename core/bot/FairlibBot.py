from pathlib import Path

from core.bot.BaseSC2Bot import BaseSC2Bot
from core import console

from fairlib import (
    OllamaAdapter,
    ToolRegistry,
    ToolExecutor,
    WorkingMemory,
    ReActPlanner,
    SimpleAgent,
)

OLLAMA_HOST = "http://localhost:11434"

_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompt.txt"
_SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")


class FairlibBot(BaseSC2Bot):
    """
    SC2 bot that routes LLM calls through the fairlib agent stack.
    Uses an empty ToolRegistry (no tools) so the agent passes straight
    through to the LLM without a ReAct tool loop.
    """

    name: str = "FairlibBot"

    def __init__(self):
        super().__init__()

        llm = OllamaAdapter(self.MODEL_NAME, host=OLLAMA_HOST)

        tool_registry = ToolRegistry()
        executor = ToolExecutor(tool_registry)
        memory = WorkingMemory()
        planner = ReActPlanner(llm, tool_registry)

        self._agent = SimpleAgent(
            llm=llm,
            planner=planner,
            tool_executor=executor,
            memory=memory,
            stateless=True,
        )

    async def get_new_directive_async(self, current_battlefield_obs: str, step: int = 0) -> str:
        full_prompt = _SYSTEM_PROMPT + current_battlefield_obs
        console.print_llm_prompt(step, self.MODEL_NAME, full_prompt)
        return await self._agent.arun(full_prompt)
