from langchain.agents import create_agent, AgentState
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolRuntime

from model_identifiers import qwen2dot5_7b_identifier


def get_user_info(runtime: ToolRuntime) -> str:
    """Look up information about the current user."""
    user_id = runtime.state.get("user_id")
    preferences = runtime.state.get("preferences")
    if not user_id:
        return "No user_id found in state."
    return f"User {user_id} preferences: {preferences}"


class CustomAgentState(AgentState):
    user_id: str
    preferences: dict


agent = create_agent(
    qwen2dot5_7b_identifier,
    tools=[get_user_info],
    state_schema=CustomAgentState,
    checkpointer=InMemorySaver(),
)

thread_config = {"configurable": {"thread_id": "1"}}

# Custom state can be passed in invoke
response = agent.invoke(
    input={
        "messages": [{"role": "user", "content": "Hello"}],
        "user_id": "user_123",
        "preferences": {"theme": "dark"}
    },
    config=thread_config
)
print("==" * 60)
print(response["messages"][-1].content)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "调用get_user_info查询用户user_123的preferences是什么？"}]},
    thread_config,
)
print("==" * 60)
print(response["messages"][-1].content)
