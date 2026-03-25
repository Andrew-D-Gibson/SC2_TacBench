from BaseSC2Bot import BaseSC2Bot

from fairlib import (
    OllamaAdapter,
    ToolRegistry,
    ToolExecutor,
    WorkingMemory,
    ReActPlanner,
    SimpleAgent,
)

OLLAMA_HOST = "http://localhost:11434"

DIRECTIVE_PROMPT = (
    'You are a StarCraft 2 tactical AI. Respond with ONLY a JSON object, no other text:\n'
    '{"directive": "FOCUS_FIRE", "reasoning": "[FairlibBot] Focus fire on the weakest enemy unit."}'
)


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

    async def get_new_directive_async(self, current_battlefield_obs: str) -> str:
        return await self._agent.arun(DIRECTIVE_PROMPT)
