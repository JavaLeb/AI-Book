from langchain.tools import tool
from models import qwen2dot5_7b_model as model


@tool
def get_weather(location: str) -> str:
    """Get the weather at a location."""
    return f"It's sunny in {location}."


model_with_tools = model.bind_tools([get_weather])

for chunk in model_with_tools.stream("What's the weather in Boston and Tokyo?"):
    # Tool call chunks arrive progressively
    for tool_chunk in chunk.tool_call_chunks:
        print(tool_chunk)
        if name := tool_chunk.get("name"):
            print(f"Tool: {name}")
        if id_ := tool_chunk.get("id"):
            print(f"ID: {id_}")
        if args := tool_chunk.get("args"):
            print(f"Args: {args}")

gathered = None
for chunk in model_with_tools.stream("What's the weather in Boston?"):
    gathered = chunk if gathered is None else gathered + chunk
    print(gathered.tool_calls)