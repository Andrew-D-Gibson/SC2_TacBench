import asyncio
from fairlib import (
    settings,
    OllamaAdapter,
    ToolRegistry,
    SafeCalculatorTool,
    ToolExecutor,
    WorkingMemory,
    ReActPlanner,
    SimpleAgent
)

async def main():
    print("Initializing a single agent...")

    # 1. The "Brain": Initialize the LLM adapter from the settings file
    llm = OllamaAdapter("gemma3:12b", host="http://localhost:11434")
    # llm = OpenAIAdapter(
    #     api_key=settings.api_keys.openai_api_key,
    #     model_name=settings.models["openai_gpt4"].model_name
    # )

    # 2. The "Toolbelt": Create a registry and add tools
    tool_registry = ToolRegistry()
    tool_registry.register_tool(SafeCalculatorTool())

    # 3. The "Hands": Create an executor that uses the toolbelt
    executor = ToolExecutor(tool_registry)

    # 4. The "Memory": Set up short-term memory for the conversation
    memory = WorkingMemory()

    # 5. The "Mind": Create the planner that uses the brain and tools
    planner = ReActPlanner(llm, tool_registry)

    # 6. Assemble the Agent: Combine all parts into a functional unit
    agent = SimpleAgent(
        llm=llm,
        planner=planner,
        tool_executor=executor,
        memory=memory,
        stateless=True
    )
    print("✅ Agent created. Ask a math question or type 'exit'.")

    # 7. Run the agent in a loop
    while True:
        try:
            user_input = input("\n👤 You: ")
            if user_input.lower() == "exit":
                break
            response = await agent.arun(user_input)
            print(f"🤖 Agent: {response}")
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    asyncio.run(main())