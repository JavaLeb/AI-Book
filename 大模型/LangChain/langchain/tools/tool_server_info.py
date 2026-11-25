from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from models import qwen2dot5_7b_model as model


@tool
def get_assistant_scoped_data(runtime: ToolRuntime) -> str:
    """Fetch data scoped to the current assistant."""

    server = runtime.server_info
    print('server=', server)

    if server is not None:
        print(f"Assistant: {server.assistant_id}, Graph: {server.graph_id}")
        if server.user is not None:
            print(f"User: {server.user.identity}")

    return "done"


agent = create_agent(
    model,
    tools=[get_assistant_scoped_data]
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "请调用 get_assistant_scoped_data 工具"}]}
)

for message in result["messages"]:
    message.pretty_print()
