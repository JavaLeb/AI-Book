from langchain.tools import tool
from models import qwen2dot5_7b_model as model


@tool
def get_weather(location: str) -> str:
    """Get the weather at a location."""
    return f"It's sunny in {location}."


# Bind (potentially multiple) tools to the model
tools = [get_weather]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)

# Step 1: Model generates tool calls
messages = [{"role": "user", "content": "What's the weather in Boston?"}]
ai_msg = model_with_tools.invoke(messages)
messages.append(ai_msg)

# Step 2: Execute tools and collect results
for tool_call in ai_msg.tool_calls:
    print(f"type of tool_call={type(tool_call)}, tool_call={tool_call}")
    # Execute the tool with the generated arguments
    tool = tools_by_name[tool_call["name"]]
    print(f'type of tool={type(tool)}')
    tool_result = tool.invoke(tool_call)
    print(f'type of tool_result={type(tool_result)}')
    messages.append(tool_result)

# Step 3: Pass results back to model for final response
for message in messages:
    print(f'type of message={type(message)},message={message}')
final_response = model_with_tools.invoke(messages)
print(f'final text={final_response.text}')
