from models import qwen2dot5_7b_model as model
from pprint import pprint


def get_weather(location: str) -> str:
    """Get the weather at a location."""
    ...


model_with_tools = model.bind_tools([get_weather])
response = model_with_tools.invoke("What's the weather in Paris?")
print(type(response))  # <class 'langchain.messages.AIMessage'>

print('--------------------------------------------')
pprint(response.model_dump())

print('--------------------------------------------')
print(f'response.usage_metadata={response.usage_metadata}')

print('--------------------------------------------')
for tool_call in response.tool_calls:
    print(f"Tool: {tool_call['name']}")
    print(f"Args: {tool_call['args']}")
    print(f"ID: {tool_call['id']}")

print('--------------------------------------------')
chunks = []
full_message = None
for chunk in model.stream("Hi"):
    chunks.append(chunk)
    print(chunk.text)
    full_message = chunk if full_message is None else full_message + chunk
