from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

@tool
def log_execution_context(runtime: ToolRuntime) -> str:
    """Log execution identity information."""

    info = runtime.execution_info
    print(f"Thread: {info.thread_id}, Run: {info.run_id}")
    print(f"Attempt: {info.node_attempt}")

    return "done"


# 使用本地模型演示，例如 Ollama
model = init_chat_model(model="ollama:qwen2.5:7b")

agent = create_agent(
    model=model,
    tools=[log_execution_context]
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "请调用 log_execution_context 工具"}]},
    config={"configurable": {"thread_id": "demo-thread-1"}},
)

for message in result["messages"]:
    message.pretty_print()