from langchain.tools import tool
from models import qwen2dot5_7b_model as model


@tool
def get_weather(location: str) -> str:
    """Get the weather at a location."""
    return f"It's sunny in {location}."


tools = [get_weather]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)

response = model_with_tools.invoke(
    "What's the weather in Boston and Tokyo?"
)

# The model may generate multiple tool calls
print(response.tool_calls)
# [
#   {'name': 'get_weather', 'args': {'location': 'Boston'}, 'id': 'call_1'},
#   {'name': 'get_weather', 'args': {'location': 'Tokyo'}, 'id': 'call_2'},
# ]


# Execute all tools (can be done in parallel with async)
results = []
for tool_call in response.tool_calls:
    tool = tools_by_name[tool_call["name"]]
    result = tool.invoke(tool_call)
    print(f'{result}')
    results.append(result)

# 并行调用.
results = get_weather.batch([call["args"] for call in response.tool_calls])
print(results)
