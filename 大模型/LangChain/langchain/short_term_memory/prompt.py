from langchain.agents import create_agent
from typing import TypedDict
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from model_identifiers import qwen2dot5_7b_identifier

class CustomContext(TypedDict):
    user_name: str


def get_weather(city: str) -> str:
    """Get the weather in a city."""

    return f"The weather in {city} is always sunny!"


@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
    user_name = request.runtime.context["user_name"]
    system_prompt = f"打招呼格式为：Hi {user_name}."
    return system_prompt


agent = create_agent(
    model=qwen2dot5_7b_identifier,
    tools=[get_weather],
    middleware=[dynamic_system_prompt],
    context_schema=CustomContext,
)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "请先向用户打招呼，然后告诉他SF的天气情况 "}]},
    context=CustomContext(user_name="John Smith"),
)
for msg in response["messages"]:
    msg.pretty_print()