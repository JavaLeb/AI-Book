from langchain.agents import create_agent


def get_weather(city: str) -> str:
    """Get weather for a given city."""

    print("正在调用get_weather工具……")

    return f"It's always sunny in {city}!"


agent = create_agent(
    model="ollama:qwen2.5:3b-instruct",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

cities = ['beijing', 'San Francisco']
for city in cities:
    result = agent.invoke(
        input={"messages": [{"role": "user", "content": "What's the weather in " + city + "?"}]}
    )
    print(result["messages"][-1].content_blocks)
